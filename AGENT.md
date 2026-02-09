# Phase 0: Reservoir-FNO (Darcy 2D) 実装仕様

このリポジトリに、**Reservoir 化した Fourier Neural Operator (Reservoir-FNO / RFNO)** を追加し、
**通常のFNO（学習あり）** と **Reservoir-FNO（バックボーン固定＋readoutのみ ridge）** を比較できるようにする。

## スコープ（今回やらないこと）
- ストリーミング更新（RLS/忘却係数）・オンライン自己校正
- 光学4f系 / 強度検出 / 複素場観測 / ハードウェア非理想
- PINO 等の PDE 残差（教師なし）最適化

---

## 1. 問題設定（Darcy flow）

領域を 2D ユニット正方形とする:
- Ω = (0, 1)^2

入力（係数場 / 透水係数 / 拡散係数）:
- a(x) > 0

出力（圧力場）:
- u(x)

PDE（Dirichlet境界）:
-  -∇ · ( a(x) ∇u(x) ) = f(x)  in Ω
-  u(x) = 0  on ∂Ω

ここでは既存FNOのベンチに合わせ、既定で f(x) = 1 を想定する。

---

## 2. 離散化

格子サイズを S×S とする（S は downsample 後のサイズ）。
格子点:
- x_i = i h, y_j = j h,  i,j = 0,...,S-1
- h = 1/(S-1)

離散係数:
- a[i,j] ≈ a(x_i, y_j)
解:
- u[i,j] ≈ u(x_i, y_j)

### 2.1 5点差分（参考：生成データ用）
内部点 (1 ≤ i,j ≤ S-2) でフラックス型の差分（ヘテロ係数に強い）を使う。

面（セル境界）の係数は調和平均:
- a_{i+1/2,j} = 2 a_{i,j} a_{i+1,j} / (a_{i,j} + a_{i+1,j})
- a_{i-1/2,j} = 2 a_{i,j} a_{i-1,j} / (a_{i,j} + a_{i-1,j})
- 同様に a_{i,j±1/2}

離散方程式:
( a_{i+1/2,j} + a_{i-1/2,j} + a_{i,j+1/2} + a_{i,j-1/2} ) u_{i,j}
 - a_{i+1/2,j} u_{i+1,j} - a_{i-1/2,j} u_{i-1,j}
 - a_{i,j+1/2} u_{i,j+1} - a_{i,j-1/2} u_{i,j-1}
 = h^2 f_{i,j}

境界は Dirichlet u=0 として未知数から除外して解く。

---

## 3. Standard FNO2d の完全定義（実装と一致）

本リポジトリの `fourier_2d.py` に合わせた定義（padding, MLP per layer, GELU）を書く。

### 3.1 入力テンソル
入力係数場（正規化後）を a ∈ R^{S×S×1} とし、座標グリッドを concat して
- x_in[i,j] = ( a[i,j], x_i, y_j ) ∈ R^3
すなわち x_in ∈ R^{S×S×3}。

### 3.2 リフト（pointwise線形）
幅（チャネル数）を C (=width) とする。線形写像 P: R^3→R^C。
- v_0[i,j] = P( x_in[i,j] ) ∈ R^C

実装上は `nn.Linear(3, C)` を (i,j) ごとに適用。

### 3.3 2D DFT（離散フーリエ変換）
チャネルごとに 2D FFT を適用する。
v ∈ R^{C×S×S} として（バッチ次元は省略）、
- \hat{v}[c, k1, k2] = Σ_{i=0}^{S-1} Σ_{j=0}^{S-1} v[c,i,j] exp( -2π i ( i k1 / S + j k2 / S ) )

実装は `torch.fft.rfft2`（最後の次元は実数→半分の複素周波数）を使う。

### 3.4 スペクトル畳み込み（低周波モードのみ学習/適用）
モード数を m1, m2 とする（既定では modes=m1=m2）。
学習パラメータは複素テンソル:
- W1 ∈ C^{C_in×C_out×m1×m2}
- W2 ∈ C^{C_in×C_out×m1×m2}

入力のフーリエ係数 \hat{v} に対し、
- out_ft[:, :, 0:m1, 0:m2]      = einsum( \hat{v}[:, :, 0:m1, 0:m2], W1 )
- out_ft[:, :, -m1:, 0:m2]     = einsum( \hat{v}[:, :, -m1:, 0:m2], W2 )
- その他の周波数は 0

その後、逆変換:
- K(v) = irfft2(out_ft)

これが `SpectralConv2d` に一致する。

### 3.5 1層の更新（リポジトリ実装に合わせた形）
層 ℓ=0..L-1 で（ここでは L=4）、
- z1 = SpectralConv2d_ℓ( v_ℓ )          （非局所：フーリエ空間で線形変換）
- z1 = MLP_ℓ( z1 )                      （1×1 conv → GELU → 1×1 conv）
- z2 = W_ℓ( v_ℓ )                       （1×1 conv）
- v_{ℓ+1} = GELU( z1 + z2 )             （最後の層だけは GELU を入れない実装もあるが、元実装に従う）

さらに入力が非周期のため padding を入れる:
- v_0 を (S+pad)×(S+pad) に zero-pad して畳み込みを回し、最後に pad を切り落とす。

### 3.6 出力射影（pointwise MLP）
最終表現 v_L ∈ R^{C×S×S} に対し、pointwise MLP Q: R^C→R を適用して
- \hat{u}[i,j] = Q( v_L[:,i,j] ) ∈ R
実装は `MLP(C, 1, 4C)`（1×1 conv→GELU→1×1 conv）。

---

## 4. Reservoir-FNO（RFNO）の完全定義

**Reservoir 化**の定義（Phase 0）:
- Standard FNO の **バックボーン（P, SpectralConv, MLP, W など）をランダム初期化して固定**する。
- **学習するのは readout のみ**。
- readout 学習は **ridge 回帰（閉形式）**で行う。

### 4.1 固定バックボーンによる特徴写像
ランダム初期化した固定パラメータ θ_res を持つ FNO バックボーンを
- Φ_{θ_res}: a ↦ v_L
と書く。v_L[i,j] ∈ R^C を各点の特徴とみなす。

### 4.2 readout（空間共有の線形射影）
readout パラメータは w∈R^C, b∈R。
各格子点で
- \hat{u}[i,j] = w^T v_L[:,i,j] + b

これは `nn.Conv2d(C,1,kernel_size=1,bias=True)` と等価。

### 4.3 ridge 回帰（目的関数）
学習データ {(a^n, u^n)}_{n=1..Ntrain} に対し、
全点 (i,j) をまとめた目的関数:
J(w,b) = Σ_{n=1}^{Ntrain} Σ_{i=0}^{S-1} Σ_{j=0}^{S-1}
          ( w^T v_L^n[:,i,j] + b - u^n[i,j] )^2
        + λ ||w||_2^2

- λ ≥ 0 は ridge 係数（`--ridge-lambda`）。
- b は通常正則化しないので、中心化で扱う（下式）。

### 4.4 閉形式解（中心化による b 非正則化）
全サンプル・全点の総数:
- M = Ntrain * S * S

特徴ベクトルを φ_m ∈ R^C（m=1..M）、目的値を y_m ∈ R とする（格子をフラット化）。

平均:
- μ_φ = (1/M) Σ_m φ_m   ∈ R^C
- μ_y = (1/M) Σ_m y_m   ∈ R

中心化:
- φ'_m = φ_m - μ_φ
- y'_m = y_m - μ_y

統計量:
- S = Σ_m φ'_m (φ'_m)^T = (Σ φφ^T) - M μ_φ μ_φ^T   ∈ R^{C×C}
- t = Σ_m φ'_m y'_m     = (Σ φ y)  - M μ_φ μ_y     ∈ R^C

解:
- w = ( S + λ I )^{-1} t
- b = μ_y - μ_φ^T w

### 4.5 実装上の重要点（メモリ爆発回避）
設計行列 (M×C) を保存せずに済むよう、DataLoader の各ミニバッチで以下を累積する:

バッチ特徴 F ∈ R^{B×C×S×S} を N=S*S にreshapeして F_b ∈ R^{B×C×N}、
ターゲット y ∈ R^{B×S×S} を y_b ∈ R^{B×N} とする。

- sum_phi    += Σ_{b,n} F_b[:, :, n]            （R^C）
- sum_y      += Σ_{b,n} y_b[:, n]               （R）
- sum_phiphi += Σ_b (F_b[b] @ F_b[b]^T)         （R^{C×C}）
- sum_phiy   += Σ_{b,n} F_b[:, :, n] * y_b[:,n] （R^C）
- count      += B*N

最後に上式で μ_φ, μ_y, S, t を作って解く。

---

## 5. データ取り扱い（.mat か 生成か）

### 5.1 既存 .mat 読み込み（互換）
`fourier_2d.py` と同一:
- MatReader で 'coeff', 'sol' を読む
- shape: (N, grid_size, grid_size)
- downsample 率 r で [:, ::r, ::r] を取り、S = int(((grid_size-1)/r)+1)

### 5.2 Python 生成（オプション）
係数場生成:
- 既存 `data_generation/navier_stokes/random_fields.py` の `GaussianRF(dim=2, size=S, alpha=2, tau=3)` を使う。
- z ~ GRF をサンプル

係数マップ:
- piecewise: a = a_pos if z>=0 else a_neg （FNO論文の ψ と同型: 12 / 3）
- exp: a = exp(z)（連続）

PDE解法:
- 2.1 の 5点差分 + SciPy sparse linear solver（spsolve / cg）

出力:
- coeff: (N, S, S)
- sol:   (N, S, S)
- 任意で .mat 保存（fields は 'coeff','sol'）

---

## 6. 実装するファイル（推奨構成）

### 6.1 新規
- `scripts/compare_darcy_fno_vs_reservoir.py`
  - `--data-source {mat,generate}`
  - 通常FNO 学習 + Reservoir-FNO ridge を同一データで比較
  - 結果を `results/.../metrics.json` に保存
  - 既存 `viz_utils.py` で learning curve / サンプル可視化 / error hist を保存

- `rfno/` package（例）
  - `rfno/models_fno2d.py` : FNO2d（既存と同等） + forward_features
  - `rfno/reservoir_readout.py` : ridge 学習ユーティリティ（統計累積→解）
  - `rfno/darcy_generate.py` : 生成データ関数

- `configs/compare_darcy_small.json`
  - 小規模スモーク用（生成データ、grid小、epochs少）

### 6.2 変更（任意だが推奨）
- `README.md` に Reservoir-FNO の説明と実行例を追記
- 既存 `fourier_2d.py` は壊さない（ただし import 先変更などの軽微な整備はOK）

---

## 7. 評価指標
- 相対L2誤差: ||u - û||_2 / ||u||_2 （既存 `LpLoss` / `viz_utils.rel_l2` と整合）
- 可能なら平均・中央値の両方を metrics.json に書く

---

## 8. 実験プロトコル（Phase 0）
1. まず generate データで小規模比較（スモーク）
   - S=32, ntrain=20, ntest=5, FNO epochs=2
2. 次に .mat データで本番比較
   - fourier_2d.py と同じ split を使う
   - FNO: epochs=500 等
   - RFNO: ridge λ を sweep（例 0, 1e-6, 1e-4, 1e-2）

---

## 9. スモーク実行コマンド（必須で通す）
- `python scripts/compare_darcy_fno_vs_reservoir.py --config configs/compare_darcy_small.json`

期待:
- 実行が完走する
- 標準出力で FNO と RFNO の test 相対L2 が表示される
- results ディレクトリに metrics.json と可視化（png/pdf/svg）が生成される
