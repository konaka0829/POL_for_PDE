# AGENT.md — Koopman‑Reservoir Operator Learning (1D Burgers, backprop‑free)

## TL;DR
このリポジトリ（FNO実装）に、**逆伝播なし**（最小二乗 / ridge のみ）で学習する
**Koopman‑Reservoir Operator** を追加する。対象は **1D Burgers のみ**。
新規スクリプト `koopman_reservoir_1d.py` を追加し、既存の `fourier_1d.py` と同様に
データ読み込み（MAT）と可視化（PNG/PDF/SVG）を行う。

---

## 目的
- PDEの時間発展を **演算子（関数→関数）**として学習するが、学習は backprop を使わず、
  **ランダム固定の encoder + 線形 Koopman + 線形 decoder を ridge で一発推定**する。
- まずは Burgers の **1-step（u0→u1）**を再現できることを最重要とする。

---

## 対象データ（Burgers）
- 既存 `fourier_1d.py` と同じMATを想定
  - `a`: 初期条件 u0(x), shape = [N, S_full]
  - `u`: 目標 u1(x), shape = [N, S_full]
- `--sub` による間引き後の空間点数: `S = 2**13 // sub`（fourier_1d.py と一致）
- 追加対応（歓迎）:
  - もし `u` が [N, T, S] の時系列を含むなら、(u_t, u_{t+1}) ペアを作ってK学習できるようにする
  - ただし “1D Burgers の枠内” を超えない

---

## モデル（最小構成）
予測は以下で行う（1-step）:

1. 測定: `m = M(u)`  （u: [S] → m: [p]）
2. Reservoir encoder: `z = E(m)` （m: [p] → z: [m_res]）
3. Koopman: `z1 ≈ K z0`
4. Decoder: `u ≈ Decode(z)` （z: [m_res] → u: [S]）

推論: `u_hat1 = Decode( K @ Encode( M(u0) ) )`

---

## 重要：基底関数（複数実装する）
このタスクでは **encoder側（測定）** と **decoder側（復元）**で使う基底を複数用意する。

### 測定基底 ψ（`--measure-basis`）
基本は内積で測定:
- `m_j = <u, ψ_j> ≈ Σ_i u(x_i) ψ_j(x_i) Δx`

最低限の選択肢:
- `fourier` : 実フーリエ基底（1, cos, sin）
- `random_fourier` : ψ_j(x)=sqrt(2)cos(2π ω_j x + b_j)
- `legendre` or `chebyshev` : 多項式基底（[0,1]→[-1,1]写像）
- `rbf` : ガウスRBF（中心等間隔、幅 `--rbf-sigma`）
（任意: `sensor` = 点サンプリング）

### 復元基底 φ（`--decoder-basis`）
最低限:
- `grid` : u ≈ D z（D: [S, m_res]）
- `fourier` : u ≈ Φ c,  c = D z
- `legendre/chebyshev/rbf` のいずれか

---

## Reservoir Encoder（固定ランダム）
実装する update:
- r_{l+1} = (1-α) r_l + α tanh( W r_l + U m + b )
- z = r_{Lw}

必須CLI:
- `--reservoir-dim`, `--washout`, `--leak-alpha`
- `--spectral-radius`（Wをスケーリング）
- `--input-scale`, `--bias-scale`（任意だが推奨）

注意:
- Wのスペクトル半径調整は power iteration 等の近似でOK
- seed（`--seed`）で完全再現できるように、torch/numpy のseedを揃える

---

## Ridge 解（数値安定）
### Koopman K
目的:
- min_K ||Z1 - K Z0||_F^2 + λK ||K||_F^2

解:
- K = Z1 Z0^T (Z0 Z0^T + λK I)^{-1}

実装:
- `torch.linalg.solve` を使い、明示的 inverse を避ける
- `--ridge-k` が λK

任意（推奨）:
- `--stabilize-k` で spectral radius を推定し、必要なら K をスケールして ≤ 1 にする
  - 推定は `torch.linalg.eigvals` でも可（サイズが大きいなら power iteration）

### Decoder D
- `grid` の場合: u ≈ D z を ridge
- 基底の場合:
  1) u ≈ Φ c を ridge で解いて c を作る（係数抽出）
  2) c ≈ D z を ridge

`--ridge-d` を λD に使う。

---

## 実装ファイル方針
- 追加（必須）:
  - `koopman_reservoir_1d.py`（standalone script）
- 追加（推奨）:
  - `koopman_reservoir_utils.py`（basis生成、ridge、reservoir等を分離）
- 更新（必須）:
  - `README.md` に実行例と主要引数を追記

既存ファイル:
- `cli_utils.py` の `add_data_mode_args`, `add_split_args`, `validate_data_mode_args` を利用して
  `fourier_1d.py` と同等のCLI設計に揃える
- 可視化は `viz_utils.py` の関数を利用し、png/pdf/svg を保存する

---

## 可視化（必須）
出力先:
- `visualizations/koopman_reservoir_1d/`

保存するもの:
- test相対L2 ヒストグラム（`plot_error_histogram`）
- 代表サンプル3件の1D比較（`plot_1d_prediction`）
  - input(u0), GT(u1), Pred(u_hat1)

---

## CLI 要件（最低限）
- data:
  - `--data-mode {single_split,separate_files}`
  - `--data-file`, `--train-file`, `--test-file`
  - `--ntrain`, `--ntest`, `--sub`, `--train-split`, `--seed`, `--shuffle`
- model:
  - `--measure-basis`, `--measure-dim`
  - `--decoder-basis`, `--decoder-dim`
  - `--reservoir-dim`, `--washout`, `--leak-alpha`
  - `--spectral-radius`
  - `--ridge-k`, `--ridge-d`
  - `--basis-normalize`（推奨）
  - `--stabilize-k`（推奨）

---

## DoD（受け入れ条件）
- `python -m compileall .` が通る
- `python koopman_reservoir_1d.py --help` が通る
- 小規模設定（例: `--ntrain 20 --ntest 5 --sub 64`）で最後まで走り、
  `visualizations/koopman_reservoir_1d/` にpng/pdf/svgが生成される
- basisが少なくとも2系統切替可能で動く
  - 例: measure=fourier/decoder=grid
  - 例: measure=rbf/decoder=fourier
- README.md に実行例が追加される
- 依存追加なし（requirements.txtの範囲内）

---

## 参考：推奨の実行例（READMEにも載せる）
例:
```bash
python koopman_reservoir_1d.py \
  --data-mode single_split --data-file data/burgers_data_R10.mat \
  --ntrain 1000 --ntest 100 --sub 8 --seed 0 --shuffle \
  --measure-basis fourier --measure-dim 128 \
  --reservoir-dim 512 --washout 8 --leak-alpha 1.0 --spectral-radius 0.9 \
  --decoder-basis grid --decoder-dim 0 \
  --ridge-k 1e-6 --ridge-d 1e-6 \
  --basis-normalize --stabilize-k
```
# AGENT.md — Backprop‑Free Koopman‑Reservoir Operator Learning
## 対応PDE: 1D Burgers（既存） + 2D Darcy Flow（追加）

> このリポジトリは FNO の各種スクリプトに加え、backprop 無し（最小二乗 / ridge のみ）で学習する
> Koopman‑Reservoir Operator を `koopman_reservoir_1d.py`（Burgers 1D）として実装済み。
> 本タスクでは **Darcy flow（2D）でも同様に動作する backprop‑free 実装**を追加する。

---

## 0. TL;DR（今回やること）
- 追加（必須）
  - `koopman_reservoir_2d.py` を新規作成（Darcy flow 用・standalone）
  - `koopman_reservoir_utils.py` を拡張し **2D 用の基底生成 `build_basis_2d`** を追加
- 更新（必須）
  - `README.md` に `koopman_reservoir_2d.py` の実行例・主要引数を追記
- 追加機能（必須：データ無し環境でも検証できるように）
  - `koopman_reservoir_2d.py --smoke-test` を実装し、外部データ無しでも end‑to‑end で動作確認可能にする
- 既存の `koopman_reservoir_1d.py` は **壊さない**（後方互換）

---

## 1. Darcy flow データ形式（FNO と同じ）
`fourier_2d.py` と同じ MAT を想定。

- 入力: 係数場（permeability など）
  - フィールド名: `coeff`（fallback として `a` も許容して良い）
  - shape: `[N, grid, grid]`（典型: `grid=421`）
- 出力: 解（圧力場など）
  - フィールド名: `sol`（fallback として `u` も許容して良い）
  - shape: `[N, grid, grid]`

### 1.1 ダウンサンプリング（fourier_2d.py と揃える）
- `--r` により `[:, ::r, ::r]`
- `--grid-size`（デフォルト 421）を用いて
  - `s = int(((grid_size - 1) / r) + 1)`
- 実装では安全のため `[:,:s,:s]` でクロップして良い
- 学習の内部表現として 2D を flatten して
  - `S = s*s`
  - `x_flat: [N, S]`
  - `y_flat: [N, S]`

---

## 2. Darcy 用モデル（backprop‑free）
Darcy は時間発展が無い（静的な係数→解の写像）ので、BGP（Burgers）の「半群」そのものは無い。
ただしこの repo の KRO 形式を崩さず、**“1-step operator” として線形写像を学習**する。

### 2.1 予測の形（Darcy）
- 測定: `m = M(field)`（内積により `[S] -> [P]`）
- Reservoir encoder（固定ランダム）: `z = E(m)`（`[P] -> [M]`）
- “Koopman”線形写像: `z_out ≈ z_in @ Kt`
  - ここで `z_in = E(M(coeff))`
  - `z_out(target) = E(M(sol))`
- Decoder（線形）: `sol_hat = Decode(z_out_pred)`（grid または基底復元）

推論（テスト）:
- `z_in = E(M(coeff))`
- `z_out_pred = z_in @ Kt`
- `sol_hat = Decode(z_out_pred)`

> ※Darcy では K は「潜在空間での線形写像（入力埋め込み→出力埋め込み）」として機能する。
> これにより Burgers 実装と同じ “E + K + D（全部 ridge）” の枠組みで統一できる。

---

## 3. 基底（2D）— 実装要件
`koopman_reservoir_utils.py` に **2D 用 `build_basis_2d`** を追加する。

### 3.1 インターフェース（必須）
- 追加関数:
  - `build_basis_2d(basis_name: str, dim: int, x_grid: torch.Tensor, y_grid: torch.Tensor, *, normalize: bool, rbf_sigma: float, random_fourier_scale: float, seed: int) -> torch.Tensor`
- 返り値:
  - `phi`: shape `[dim, S]`（S = s*s, flatten 済み）
- `normalize=True` の場合:
  - 離散内積重み `dxdy` を用い `normalize_basis_rows(phi, dx=dxdy)` 相当で行正規化

### 3.2 対応 basis（必須：1D と同じ集合に揃える）
`--measure-basis` / `--decoder-basis` は以下の選択肢を維持する。

- `fourier`
  - 2D の実フーリエ基底（最低限: 定数 + cos/sin(2π(kx x + ky y)) を low‑freq から詰める）
  - 目標: `dim` 本埋める（不足しないように周波数ペアを列挙）
- `random_fourier`
  - `psi_j(x,y)=sqrt(2) cos(2π(ωx x + ωy y) + b)`
  - `ω ~ Normal(0, random_fourier_scale)`、`b ~ Unif(0, 2π)`
- `legendre`, `chebyshev`
  - 1D 基底（x 軸、y 軸）を作り、テンソル積で 2D を作る（lexicographic で dim まで切る）
- `rbf`
  - 2D ガウス RBF（中心は格子状に配置して dim まで）
  - `exp(-((x-cx)^2 + (y-cy)^2)/(2 sigma^2))`
- `sensor`
  - 2D グリッド上の点サンプリングを内積として表現（one‑hot を 1/dxdy でスケール）
  - センサ点は均等間隔の subgrid から dim 個取る

---

## 4. 既存 util の再利用（重要）
- `measure_with_basis(u, psi, dx)` は 2D でもそのまま使える（dx に `dxdy` を渡す）
  - 前提: `u: [N,S]`, `psi: [P,S]`
- `fit_basis_coefficients(u, phi, ridge)` は 2D でも使える（flatten 前提）
- `FixedReservoirEncoder` は 2D/1D に依存しない（m ベクトル次元のみ）

---

## 5. Darcy 用スクリプト `koopman_reservoir_2d.py` 要件
### 5.1 実装スタイル
- `fourier_2d.py` / `koopman_reservoir_1d.py` と同じく **standalone**
- 依存追加なし（`requirements.txt` の範囲内）
- `--device {auto,cpu,cuda}` 対応（既存 util の `to_device` を使用）

### 5.2 CLI（必須）
`cli_utils.py` を使って FNO と整合する CLI にする。

#### データ系
- `--data-mode {single_split,separate_files}`
- `--data-file`（single_split 時）
- `--train-file`, `--test-file`（separate_files 時）
- `--train-split`, `--seed`, `--shuffle`（single_split 時の分割）
- `--ntrain`（default=1000）
- `--ntest`（default=100）
- `--r`（default=5）
- `--grid-size`（default=421）
- `--normalize {none,unit_gaussian}`（default=unit_gaussian）
  - `unit_gaussian` の場合は `utilities3.UnitGaussianNormalizer` を使用し、予測は decode してから評価する

#### モデル系（Burgers と揃える）
- `--measure-basis {fourier,random_fourier,legendre,chebyshev,rbf,sensor}`
- `--measure-dim`（default=256 など、1D よりやや大きめが無難）
- `--decoder-basis {grid,fourier,legendre,chebyshev,rbf}`
- `--decoder-dim`（grid 以外必須）
- `--rbf-sigma`
- `--random-fourier-scale`
- `--basis-normalize`
- `--reservoir-dim`
- `--washout`
- `--leak-alpha`
- `--spectral-radius`
- `--input-scale`
- `--bias-scale`
- `--ridge-k`
- `--ridge-d`
- `--stabilize-k`
- `--device`

#### Smoke test（必須）
- `--smoke-test`（action="store_true"）
  - True の場合、外部 MAT を読まずに擬似データを生成して end‑to‑end を通す
  - デフォルトの smoke 空間サイズは小さく（例: s=33）、CPU でもすぐ終わるようにする
  - 係数→解の関係は簡単で良い（例: avg_pool2d による平滑化など）
  - smoke test でも可視化ファイルが生成されること

### 5.3 処理フロー（Darcy）
1. データ読み込み（MAT または smoke）
2. downsample（MAT の場合）
3. （任意）正規化（unit_gaussian）
4. flatten（`[N,s,s] -> [N,S]`）
5. 2D grid を作成
   - `x_grid = linspace(0,1,s)`, `y_grid = linspace(0,1,s)`
   - `dxdy = (1/(s-1))^2`
6. 測定基底 `psi = build_basis_2d(measure_basis, P, x_grid, y_grid, ...)`（shape `[P,S]`）
7. `m_in = measure_with_basis(x_flat, psi, dx=dxdy)`（`[N,P]`）
8. `m_out = measure_with_basis(y_flat, psi, dx=dxdy)`（`[N,P]`）
9. `z_in = reservoir.encode(m_in)`、`z_out = reservoir.encode(m_out)`
10. `Kt = fit_koopman(z_in, z_out, ridge_k)`（必要なら stabilize）
11. decoder 学習（**Darcy では出力用のみ**）
    - grid: `D = ridge_fit_linear_map(z_out, y_flat, ridge_d)`
    - basis: `phi = build_basis_2d(decoder_basis, Q, ...)`
      - `c = fit_basis_coefficients(y_flat, phi, ridge_d)`
      - `D = ridge_fit_linear_map(z_out, c, ridge_d)`
12. 推論:
    - `z_out_hat = z_in_test @ Kt`
    - `y_hat_flat = decode(z_out_hat)`
    - reshape +（必要なら）y_normalizer.decode
13. 評価:
    - per-sample relL2（`viz_utils.rel_l2`）
    - mean/median を print
14. 可視化保存

---

## 6. 可視化（必須）
出力先:
- `visualizations/koopman_reservoir_2d/`

保存するもの（png/pdf/svg 全部）:
- `test_relL2_hist`：テスト相対L2ヒストグラム（`plot_error_histogram`）
- `sample_000..002`：`plot_2d_comparison(gt, pred, input_field=coeff)` を少なくとも3サンプル
  - パネル: input / GT / Pred / abs_err / rel_err_map

---

## 7. README 更新（必須）
`README.md` の日本語セクションに `koopman_reservoir_2d.py` を追加し、
- 実行例（separate_files と single_split）
- 主要引数とデフォルト値
- smoke test 実行例
を追記する。

---

## 8. DoD（受け入れ条件）
- `python -m compileall .` が通る
- `python koopman_reservoir_1d.py --help` が通る（既存維持）
- `python koopman_reservoir_2d.py --help` が通る
- `python koopman_reservoir_2d.py --smoke-test --device cpu --ntrain 16 --ntest 4` が最後まで走る
  - `visualizations/koopman_reservoir_2d/` に png/pdf/svg が生成される
- 依存追加なし（requirements.txt 変更不要）
- 2D basis が少なくとも 2 系統で切り替え可能
  - 例: measure=random_fourier / decoder=grid
  - 例: measure=rbf / decoder=fourier

---

## 9. 推奨実行例（README にも載せる）
### Smoke test（データ無しで動作確認）
```bash
python koopman_reservoir_2d.py --smoke-test --device cpu \
  --ntrain 16 --ntest 4 \
  --measure-basis random_fourier --measure-dim 64 \
  --reservoir-dim 128 --washout 4 --leak-alpha 1.0 --spectral-radius 0.9 \
  --decoder-basis grid \
  --ridge-k 1e-6 --ridge-d 1e-6 \
  --basis-normalize --stabilize-k
```

### Darcy flow（separate_files）実行例
```bash
python koopman_reservoir_2d.py --data-mode separate_files \
  --train-file data/piececonst_r421_N1024_smooth1.mat \
  --test-file  data/piececonst_r421_N1024_smooth2.mat \
  --ntrain 1000 --ntest 100 --r 5 --grid-size 421 \
  --normalize unit_gaussian \
  --measure-basis random_fourier --measure-dim 256 \
  --reservoir-dim 512 --washout 8 --leak-alpha 1.0 --spectral-radius 0.9 \
  --decoder-basis grid \
  --ridge-k 1e-6 --ridge-d 1e-6 \
  --basis-normalize --stabilize-k
```

## 10. 実装上の注意（重要）
- 明示的 inverse は避け、`torch.linalg.solve` を使う（既存 util と同じ）
- `build_basis_2d` は GPU 上でも動くよう torch 演算で書く（乱数は numpy RNG でも可だが device/dtype に注意）
- 2D flatten の reshape を間違えない（`[N,s,s] <-> [N,S]`）
- v7.3 MAT は `utilities3.MatReader` で読める
- 可視化は `viz_utils.save_figure_all_formats` 既存仕様（png/pdf/svg）に準拠する
# AGENT.md — 時系列PDEデータ生成 + FNO/KRO 多段ステップ予測ベンチマーク拡張

## 0. 目的（TL;DR）
このリポジトリは FNO / KRO を複数PDEに対して実行できるが、現状は主に **入力→出力（1-shot）学習**であり、特に **KRO（Koopman-Reservoir Operator）が「K を繰り返し適用して多段ステップ予測できる」強み**を活かし切れていない。

本タスクでは、**適切な 1D/2D PDE を選定し、時系列データ（複数初期条件の軌道）を生成し、FNO と KRO の両方で多段ステップ予測（rollout）を評価できる状態**に拡張する。

重要：
- **データ分割は必ず「軌道（初期条件）単位」で行う**（時刻ペアを作ってからシャッフル分割すると train/test に同一軌道が混ざりリークする）
- データ形式は **time-last** に統一する：  
  - 1D: `u.shape == [N, S, T_total]`  
  - 2D: `u.shape == [N, S, S, T_total]`
- KROは `z_{t+1} ≈ z_t K` を学習し、推論は **K^k**（反復適用）で多段予測するのが本質

---

## 1. 既存リポジトリの前提（壊さないこと）
- 既存スクリプトは「standaloneで直接実行」方針。新機能もこの流儀に合わせる。
- 既存主要スクリプト：
  - `fourier_1d.py`：1D（静的）Burgers（t=0→1）
  - `fourier_2d.py`：2D（静的）Darcy（係数→解）
  - `fourier_2d_time.py`：2D時系列（Navier-Stokes）
  - `koopman_reservoir_1d.py`：1D（主に静的/1-step）KRO
  - `koopman_reservoir_2d.py`：2D（静的）Darcy用KRO
- これら既存スクリプトは後方互換を保つ（必要な修正は「汎用化」程度に留める）。

---

## 2. PDEの選定（慎重に）と採用理由
「多段予測」「周期境界」「スペクトル法で安定に生成できる」「1階時間発展（マルコフ）」を重視し、以下を採用する。

### 2.1 1D PDE（2本）
1) **粘性 Burgers（1D）**
- 方程式：`u_t + u u_x = ν u_xx`
- 長所：安定・基準タスクとして最適。まずここで end-to-end が回ることが最重要。

2) **Kuramoto–Sivashinsky（1D）**
- 方程式（代表）：`u_t + u u_x + u_xx + u_xxxx = 0`（周期）
- 長所：散逸だがカオス性もあり、長期予測でモデル差（KRO安定化やridge）が出る。

### 2.2 2D PDE（2本）
3) **Allen–Cahn（2D）**
- 方程式：`u_t = ε^2 Δu + u - u^3`（周期）
- 長所：スカラーで扱いやすく、NSより生成も学習も軽い。2D時系列の中難度として最適。

4) **2D Navier–Stokes（既存）**
- 既存 `data_generation/navier_stokes/ns_2d.py` を活用
- 長所：難タスク。点wiseの長期 relL2 は厳しいが、時系列オペレータ学習の代表。

---

## 3. データ仕様（最重要：リーク防止・整合性）
### 3.1 保存形式（.mat）
- 基本は `scipy.io.savemat` の v5 MAT（`MatReader` が確実に読める）
- 必須フィールド：
  - `u`：時系列解
- 任意フィールド：
  - `t`：時刻配列 `[T_total]`
  - `x`, `y`：座標（保存しても良いが学習スクリプトは生成でOK）

### 3.2 形状（統一）
- 1D：`u.shape = [N, S, T_total]`（time-last）
- 2D：`u.shape = [N, S, S, T_total]`（time-last）

### 3.3 train/test 分割（軌道単位）
絶対に守る：
1. まず `N` 本の軌道を train/test に分割する（`indices` を作る）
2. その後、train 軌道から (t→t+1) ペアを作る

NG：
- 全ての (t→t+1) ペアを作ってからシャッフル分割（同じ軌道が混ざりリーク）

---

## 4. 実装する成果物（ファイル単位のToDo）
### 4.1 新規追加：データ生成（Python）
- `data_generation/burgers/burgers_1d_time.py`
- `data_generation/ks/ks_1d.py`
- `data_generation/allen_cahn/ac_2d.py`

要件：
- CLIで N/S/T/dt/record_steps/seed 等を指定できる
- 初期条件は `data_generation/navier_stokes/random_fields.py` の `GaussianRF(dim=1/2)` を流用
- 出力 `.mat` の `u` が **time-last** になること（必須）
- 可能なら `t` も保存

### 4.2 新規追加：学習・評価スクリプト
- `fourier_1d_time.py`（新規）
- `koopman_reservoir_1d_time.py`（新規：時系列用、リーク防止分割、rollout評価）
- `koopman_reservoir_2d_time.py`（新規：時系列用、rollout評価）

### 4.3 既存修正（必要最小限の汎用化）
- `fourier_2d_time.py`：
  - 現状 `FNO2d` の入力線形層が `nn.Linear(12, width)` に固定（T_in=10前提）
  - **T_in 可変に対応**させる（`nn.Linear(T_in + 2, width)`）
  - `FNO2d` の `__init__` に `in_channels` or `T_in` を渡して決定する実装へ
  - デフォルト挙動（T_in=10）を壊さないこと

- `viz_utils.py`：
  - 1D時系列の relL2 over time を描ける関数が必要（現状は2D前提 `plot_rel_l2_over_time` が (S,S,T) 固定）
  - 新規関数例：
    - `plot_rel_l2_over_time_1d(gt: (S,T), pred: (S,T), ...)`
  - 既存関数は壊さない

- `README.md`：
  - 新規スクリプトの説明と実行例（データ生成→学習→評価）を追記

---

## 5. データ生成スクリプト仕様（詳細）
### 5.1 burgers_1d_time.py（1D Burgers）
- PDE：`u_t + u u_x = ν u_xx`
- 境界：周期
- 解法：擬スペクトル（FFT）＋時間積分（例：RK4 または 拡散を積分因子/IMEX）
  - 安定に大量生成できることが最重要（厳密高精度より「爆発しない」を優先）
- 推奨デフォルト（例。CLIで変更可能に）：
  - `S=1024`, `T_final=1.0`, `dt=1e-4`, `record_steps=200`, `nu=1e-2`
  - `N=1000`（生成時間が重いなら小さく）
- 初期条件：
  - `GaussianRF(1, S, alpha=2.0, tau=7.0)` 等をサンプルしてスケーリング（必要なら）
- 出力：
  - `u: [N, S, record_steps]`（record_steps が T_total）
  - `t: [record_steps]`

CLI例：
```bash
python data_generation/burgers/burgers_1d_time.py \
  --out data/burgers_1d_ts.mat \
  --N 200 --S 1024 --T-final 1.0 --dt 1e-4 --record-steps 200 --nu 1e-2 --seed 0
```
5.2 ks_1d.py（1D Kuramoto–Sivashinsky）

PDE：u_t + u u_x + u_xx + u_xxxx = 0（周期）

解法：ETDRK4 推奨（剛性あり）

Cox–Matthews / Kassam–Trefethen の標準ETDRK4実装でOK

推奨デフォルト例：

S=1024, T_final=50.0, dt=0.25e-2（内部dt）、record_steps=200 など

まずは爆発しないパラメータを優先し、CLIで調整可能にする

初期条件：GaussianRF（1D）

CLI例：

python data_generation/ks/ks_1d.py \
  --out data/ks_1d_ts.mat \
  --N 200 --S 1024 --T-final 50.0 --dt 2.5e-3 --record-steps 200 --seed 0

5.3 ac_2d.py（2D Allen–Cahn）

PDE：u_t = ε^2 Δu + u - u^3（周期）

解法：スペクトル＋ETDRK4 or IMEX（どちらでも可。まず安定重視）

推奨デフォルト例：

S=64（FNO2d_time の標準と整合）

T_final=1.0, dt=1e-3, record_steps=200, epsilon=0.01 など

初期条件：GaussianRF(2, S, alpha=2.5, tau=7) など

CLI例：

python data_generation/allen_cahn/ac_2d.py \
  --out data/allen_cahn_2d_ts.mat \
  --N 200 --S 64 --T-final 1.0 --dt 1e-3 --record-steps 200 --epsilon 0.01 --seed 0

6. 学習・評価スクリプト仕様（詳細）
6.1 fourier_1d_time.py（新規：1D 時系列FNO）

目的：

入力：過去 T_in ステップ（基本 T_in=1）から次の u_{t+1} を予測する演算子を学習

推論：自己回帰で T ステップ先まで rollout

設計方針：

fourier_2d_time.py の構造を踏襲し 1D 化する

FNO層：SpectralConv1d

入力チャネル：T_in + 1（過去T_in + x座標）

出力チャネル：1（u）

CLI要件（最低限）：

data-mode / split は cli_utils.py に揃える

--sub, --S, --T-in, --T, --step, --epochs, --modes, --width, --batch-size, --learning-rate

できれば --field（読み込むMATキー、デフォルト u）

データ読み込み：

MatReader で u を読み、1Dなので u[train_idx, ::sub, :T_in] などで切る

形状は train_a: [ntrain, S, T_in], train_u: [ntrain, S, T]

訓練ループ：

fourier_2d_time.py と同様に

for t in range(0,T,step):

y = yy[..., t:t+step]

im = model(xx)

loss += ...

xx = concat(xx[..., step:], im)

可視化：

学習曲線（viz_utils）

test サンプルの 1D 予測プロット（数枚）

test の relL2 ヒストグラム

最小実行例：

python fourier_1d_time.py --data-mode single_split --data-file data/burgers_1d_ts.mat \
  --ntrain 100 --ntest 20 --sub 1 --S 1024 --T-in 1 --T 50 --step 1 --epochs 10

6.2 fourier_2d_time.py（既存修正：T_in可変化）

現状問題：

FNO2d.p = nn.Linear(12, width) 固定で T_in=10 を暗黙に要求している

必須修正：

FNO2d.__init__ に in_channels（= T_in + 2）を渡す

self.p = nn.Linear(in_channels, width) にする

スクリプト側で in_channels = args.T_in + 2 を計算して渡す

デフォルト T_in=10 は同一挙動になるようにする

これにより Allen–Cahn などでも --T-in 1 を指定して「1-step演算子」学習が可能になる。

6.3 koopman_reservoir_1d_time.py（新規：KRO 時系列・多段予測）

目的：

1-step の Koopman 行列 K を ridge で学習し、推論で K^k による rollout を行う

軌道分割でリーク防止し、per-step誤差カーブを出す

モデル（既存utilsを利用）：

測定：m_t = <u_t, ψ> （build_basis_1d, measure_with_basis）

固定reservoir encoder：z_t = E(m_t) （FixedReservoirEncoder）

Koopman：z_{t+1} ≈ z_t @ K （fit_koopman）

Decoder：

まずは grid decoder：u_t ≈ z_t @ D（ridge_fit_linear_map）

既存の basis decoder も選べるように（decoder-basis）

重要：Decoder の学習は 全時刻で行う

train trajectories の全 (u_t) について z_t を作り、z_t -> u_t を ridge で学習
（1-step ペアだけでdecoderを学習すると rollout 時に崩れやすい）

データ読み込み：

MAT から u を読む（[N,S,T_total]）

single_split なら indices を作って 軌道単位で分割

separate_files なら train/test ファイルの軌道をそのまま使用

--ntrain/--ntest は軌道数を指す（ペア数ではない）

学習用ペア作成：

train軌道のみから

x_pairs = u_train[..., 0:T_total-1]

y_pairs = u_train[..., 1:T_total]

ただし内部表現は [Npairs, S] に reshape して basis/encoder へ

rollout評価：

各 test 軌道 i について

初期 u0 = u_i[..., t0]（通常 t0 = T_in-1 だが Markov なら 0）

z = encode(measure(u0))

for k=1..T_pred:

z = z @ K

u_hat_k = decode(z)

u_gt_k = u_i[..., t0+k]

出力：

per-step relL2 の平均曲線

全軌道平均の full-trajectory relL2

サンプル可視化（いくつかの時刻で 1D曲線を比較）

CLI要件：

cli_utils の data-mode/split と整合

追加：

--T（予測ホライズン）

--t0（開始時刻 index, default 0）

--normalize {none,unit_gaussian}（任意だが推奨）

unit_gaussian は train の (N*T) を束ねて mean/std を計算し、時間に依存しない正規化にする

既存 KRO1D と同様の basis/reservoir/ridge/stabilize/device を持つ

可視化（png/pdf/svgで保存）：

visualizations/koopman_reservoir_1d_time/

test_relL2_hist.*

relL2_over_time.*（1D対応関数が必要）

代表サンプルの複数時刻プロット（例 t=0, mid, last）

smoke test：

--smoke-test を用意し、外部MAT無しで

合成時系列（例：単純な線形システムや滑らかな移流）を生成して end-to-end を確認できるようにする

6.4 koopman_reservoir_2d_time.py（新規：2D時系列KRO）

目的：

2D スカラー場の時系列に対して同様に K を学習し K^k rollout

データ：

u: [N,S,S,T_total]

処理：

flatten：u_t -> [S*S]

basis：build_basis_2d（psi: [P, S*S]）

measurement：m_t = u_vec @ (psi^T * dxdy)

encode/koopman/decoder は 1D と同様

CLI：

1D_time と同等 + --S（grid size）

--measure-basis は 2D対応のもの（既存 build_basis_2d の choices）

可視化：

plot_2d_time_slices（既存）

plot_error_histogram

plot_rel_l2_over_time（2Dは既存のまま使える）

smoke test：

2D 合成データ（例：拡散方程式でOK）で end-to-end を走らせる

7. 評価指標（最低限）

全スクリプト共通で最低限以下を出す：

per-step relL2（時間 index に対する平均）

full-trajectory relL2（全時刻まとめた相対L2）

代表サンプル可視化（1D曲線 or 2Dヒートマップの時刻スライス）

注意（NS）：

長期 pointwise relL2 は破綻しやすい。将来的に統計量（エネルギースペクトル等）を追加しても良いが、初期実装は relL2 で十分。

8. README 更新内容（必須）

新規 generator の説明とコマンド例

新規 fourier_1d_time.py, koopman_reservoir_*_time.py の実行例

「時系列データの形状」「軌道分割が必須である理由（リーク防止）」を明記

9. DoD（受け入れ条件）

python -m compileall . が通る

新規/修正スクリプトで --help が通る

小規模設定で end-to-end が通る（CPUで良い）

generator が .mat を出力

KRO_time が読み込み→学習→rollout→可視化まで完走

FNO_time は最小 epoch（例 1〜2）で完走

可視化は必ず png/pdf/svg を保存（viz_utils.save_figure_all_formats を利用）

既存 fourier_1d.py, fourier_2d.py, fourier_2d_time.py（T_in=10）、koopman_reservoir_1d.py, koopman_reservoir_2d.py が壊れない

10. 最短の動作確認フロー（例）
10.1 1D Burgers（KRO_time）
python data_generation/burgers/burgers_1d_time.py --out data/burgers_1d_ts_small.mat --N 32 --S 256 --T-final 1.0 --dt 1e-4 --record-steps 50 --nu 1e-2 --seed 0

python koopman_reservoir_1d_time.py --data-mode single_split --data-file data/burgers_1d_ts_small.mat \
  --ntrain 24 --ntest 8 --sub 1 --T 30 --t0 0 --seed 0 --shuffle \
  --measure-basis fourier --measure-dim 64 \
  --reservoir-dim 128 --washout 4 --leak-alpha 1.0 --spectral-radius 0.9 \
  --decoder-basis grid \
  --ridge-k 1e-6 --ridge-d 1e-6 --basis-normalize --stabilize-k

10.2 2D Allen–Cahn（FNO2d_time）
python data_generation/allen_cahn/ac_2d.py --out data/allen_cahn_2d_ts_small.mat --N 20 --S 64 --T-final 1.0 --dt 1e-3 --record-steps 60 --epsilon 0.01 --seed 0

python fourier_2d_time.py --data-mode single_split --data-file data/allen_cahn_2d_ts_small.mat \
  --ntrain 16 --ntest 4 --sub 1 --S 64 --T-in 1 --T 40 --epochs 2
