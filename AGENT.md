# Phase 1: PS‑RSDNO（Physics‑Shaped Random Spectral Dictionary Neural Operator）実装仕様（Darcy 2D）

本フェーズでは、既存の **Reservoir‑FNO (RFNO: ランダムFNO + ridge readout)** に対して、
**スペクトル重みを “物理整形” した辞書で生成**することで性能改善を狙う
**PS‑RSDNO** を追加実装する。

目的:
- FNO（学習あり）/ RFNO（完全ランダム）/ PS‑RSDNO（物理整形ランダム）を同一条件で比較できるようにする。
- PS‑RSDNO はバックボーン固定、readout は ridge（閉形式）で学習する。
- 特に Darcy（線形楕円型PDE）で、完全ランダムより “物理形状” に寄せた辞書が効くことを検証する。

---

## 0. 既存実装との関係

- FNO2d バックボーンは `rfno/models_fno2d.py` の `FNO2d` を基準とする。
- RFNO は `FNO2d` をランダム初期化して固定し、`RidgeReadout2D` で readout を閉形式学習する。
- 本フェーズは **SpectralConv2d の重み生成のみを置き換える**（他は基本同じ形）。

---

## 1. 問題設定（Darcy flow）

領域: Ω = (0,1)^2

入力係数場 a(x) > 0
出力 u(x)

PDE（Dirichlet）:
-  -∇·(a(x)∇u(x)) = f(x)  in Ω
-  u(x) = 0 on ∂Ω

ベンチに合わせ既定は f(x)=1。

---

## 2. 離散化

格子 S×S,  i,j=0..S-1,  h=1/(S-1)

a[i,j] ≈ a(x_i,y_j)
u[i,j] ≈ u(x_i,y_j)

（データ生成の差分は Phase0 の 5点差分仕様に従う）

---

## 3. PS‑RSDNO の“完全定義”（数式）

### 3.1 入力のリフト（FNO互換）

入力テンソル:
- a ∈ R^{S×S×1}

座標を concat:
- x_in[i,j] = (a[i,j], x_i, y_j) ∈ R^3
- x_in ∈ R^{S×S×3}

点wise線形リフト P: R^3 → R^C（C=width）:
- v0[i,j] = P(x_in[i,j]) ∈ R^C

実装: nn.Linear(3,C) を (i,j) ごとに適用し、(B,C,S,S) へ permute。

padding（非周期境界対策）:
- v を (S+pad)×(S+pad) に 0-pad し、最後に切り落とす（FNOと同じ）。

---

### 3.2 PS‑RSD（Physics‑Shaped Random Spectral Dictionary）によるスペクトル畳み込み

#### 3.2.1 rfft2 と低周波ブロック

v ∈ R^{B×C_in×S×S}

2D rfft:
- \hat v = rfft2(v) ∈ C^{B×C_in×S×(S/2+1)}

FNO と同様に “角の低周波ブロック” だけを使う:
- kx ブロック: 0..m1-1 と -m1..-1（実装上は [:m1] と [-m1:]）
- ky ブロック: 0..m2-1（実装上は [:m2]）
（ここで m1=m2=modes を基本）

---

#### 3.2.2 辞書要素 ψ_d(k)（物理整形）

Darcy は定数係数近似すると -āΔu = f であり、Fourier では
- (ā |k|^2) û(k) = \hat f(k)
- û(k) = (1/(ā|k|^2)) \hat f(k)

すなわち **inverse Laplacian / resolvent** が “物理的な核” になる。
そこで辞書を以下の **Laplacian resolvent 族**で与える。

離散周波数:
- k = (kx,ky)
- κ(k) = kx^2 + ky^2

辞書サイズ D を固定し、d=1..D についてランダムパラメータをサンプル:
- α_d ∈ (0,∞),  β_d ∈ (0,∞)
- p_d ∈ {0,1}, q_d ∈ {0,1}  （微分次数）
- s_d ∈ {0,1}               （resolvent次数; s=0 は “微分だけ/恒等”）

辞書要素（複素スカラ）:
ψ_d(kx,ky) = ( i*kx )^{p_d} ( i*ky )^{q_d} / ( α_d + β_d κ(k) )^{s_d}

- (p_d=q_d=s_d=0) は恒等(=1)
- (s_d=1, p=q=0) は smoothing（inverse Helmholtz/Laplace 型）
- (p=1,q=0,s=1) などは “勾配＋スムージング” で、フラックス/勾配を表す特徴を作る

数値安定化（推奨: 実装すること）:
低周波ブロック上で RMS 正規化する:
- rms_d = sqrt( mean_{kx,ky in block} |ψ_d(kx,ky)|^2 ) + eps
- ψ_d ← ψ_d / rms_d
（eps は 1e-12 程度）

---

#### 3.2.3 チャネル混合と重みテンソルの生成（辞書展開）

各辞書要素に対し、チャネル混合行列（複素）をランダムに固定生成:
- A_d ∈ C^{C_in×C_out}

ここで **周波数依存は ψ_d(k) のみ**に押し込み、
各周波数での線形変換行列を
- W(k) = Σ_{d=1..D} ψ_d(k) A_d

と定義する。

FNO の実装形式に合わせ、2ブロック weights1/weights2 を構成する:
- weights1[:,:,kx,ky] = W(kx,ky)     for kx in {0..m1-1}, ky in {0..m2-1}
- weights2[:,:,kx,ky] = W(kx,ky)     for kx in {-m1..-1}, ky in {0..m2-1}

注意: weights2 用の kx は符号付きで扱う（例: -1,-2,...）。実装では
- kx_neg = -(m1 - idx)  または  kx = idx - m1  のように signed 化して ψ を計算する。

最終的なスペクトル畳み込み:
- out_ft は 0 初期化
- out_ft[:,:, :m1, :m2]   = einsum( x_ft[:,:, :m1, :m2], weights1 )
- out_ft[:,:, -m1:, :m2]  = einsum( x_ft[:,:, -m1:, :m2], weights2 )
- y = irfft2(out_ft, s=(S,S))

この層を `PhysicsShapedSpectralConv2d` として実装する。
weights1/weights2 は **学習しない**ため nn.Parameter ではなく register_buffer にすること。

---

### 3.3 PS‑RSDNO2d のネットワーク構造（FNO互換）

FNO2d と同じ 4層構造を踏襲する（比較公平性のため）。
層 ℓ=0..3 で:

- z1 = PS_SpectralConv_ℓ(v_ℓ)
- z1 = MLP_ℓ(z1)          （1×1 conv → activation → 1×1 conv）
- z2 = W_ℓ(v_ℓ)           （1×1 conv）
- v_{ℓ+1} = activation( spectral_gain*z1 + skip_gain*z2 )

最後に features を返す:
- Φ(a) := v_L ∈ R^{B×C×S×S}

---

### 3.4 Readout（閉形式 ridge）

空間共有の線形 readout:
- \hat u[i,j] = w^T Φ(a)[:,i,j] + b

学習は ridge 回帰:
min_{w,b} Σ_{n,i,j} ( w^T Φ(a^n)[:,i,j] + b - u^n[i,j] )^2 + λ||w||^2

解は Phase0 の中心化閉形式（`RidgeReadout2D`）をそのまま使う。

---

## 4. 実装タスク（ファイル）

### 4.1 新規追加
- rfno/models_psrsdno2d.py
  - class PhysicsShapedSpectralConv2d
  - class PSRSDNO2d (FNO2d と同じインターフェースで forward_features を持つ)
  - 辞書サンプル関数（log-uniform で α,β を引く等）
  - 重み生成は register_buffer で固定

### 4.2 既存拡張
- scripts/compare_darcy_fno_vs_reservoir.py
  - run 選択に psrsdno を追加（例: choices に "psrsdno", "all" を追加し、"both" は後方互換）
  - PSRSDNO 用の fit_* 関数と eval_* を追加
  - metrics.json に PSRSDNO の評価結果も入れる
  - 可視化も PSRSDNO のベスト設定で出す

### 4.3 config 追加
- configs/compare_darcy_small_psrsdno.json
  - generate データの小規模スモーク
  - run: "all"（FNO/RFNO/PSRSDNO）
  - psrsdno_dict_size 等を指定

---

## 5. CLI 追加仕様（psrsdno ハイパーパラメータ）

argparse で以下を追加（dest は underscore になる）:

- --psrsdno-dict-size (int, default=32)
- --psrsdno-alpha-min (float, default=1e-1)
- --psrsdno-alpha-max (float, default=1e+1)
- --psrsdno-beta-min  (float, default=1e-2)
- --psrsdno-beta-max  (float, default=1e+0)
- --psrsdno-seed      (int, default=0)
- --psrsdno-complex-mixing (bool flag, default=True)
- --psrsdno-eps-norm  (float, default=1e-12)

辞書の (p,q,s) は固定集合から一様サンプル:
- p,q ∈ {0,1}
- s ∈ {0,1}
加えて identity を必ず入れたい場合は d=1 を強制で (p=q=s=0) とする。

---

## 6. スモーク実行（必須）

以下が完走し、results が生成されること:

python scripts/compare_darcy_fno_vs_reservoir.py --config configs/compare_darcy_small_psrsdno.json

期待:
- 標準出力に FNO/RFNO/PSRSDNO の test_relL2 が出る
- results/<run_name>/metrics.json に 3者の結果が入る
- 可視化ファイルも生成される

---

## 7. 実装上の注意

- torch.fft.rfft2 の shape は (S, S/2+1) である点に注意。
- kx の負周波（weights2）では signed kx を使って ψ_d を評価すること。
- register_buffer にした weights は .to(device) で自動で移動する。
- 再現性: torch.Generator を使い、seed を引数で制御。
- dtype: weights は torch.cfloat を推奨。readout は float64 統計（既存実装）を踏襲。
