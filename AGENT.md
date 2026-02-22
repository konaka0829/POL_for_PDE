# AGENT.md — PDE-induced Random Feature Operator Learning (PDE-RF) for this FNO repo

## 0. 目的（What to build）
このリポジトリ（FNO実装）に、研究アイデア

> **既知PDEが誘導する半群 \(T(t)=e^{tA}\) を用いたランダム特徴（RKHS）で、未知PDEの解作用素 \(G\) を回帰する**

を **1D（Burgers）** と **2D（Darcy）** の両方で動く形で追加実装する。

本実装は **FNO本体は壊さず**、同じデータローダ・可視化ユーティリティを再利用した **新しいベースラインスクリプト**として追加する：
- `pde_rf_1d.py`（Burgers: `fourier_1d.py` 相当）
- `pde_rf_2d.py`（Darcy: `fourier_2d.py` 相当）

加えて、共通ロジックをモジュール化する：
- `pde_features.py`：既知PDE半群（熱半群）とランダム特徴写像
- `ridge.py`：多出力リッジ回帰（閉形式解／Cholesky）
- `basis.py`：出力基底（最低限 `grid`。可能なら `pod` も）

**Codex CLI は会話ログを見れないため、本AGENT.md内の数式・仕様・アルゴリズム・CLI仕様を“唯一の真実”として実装すること。**


---

## 1. 数理仕様（Mathematical spec; fully-defined）

### 1.1 作用素学習の基本設定
- 入力関数空間（Hilbert）を \(H_{\text{in}}\)（離散化後は \(\mathbb{R}^{S}\) or \(\mathbb{R}^{S\times S}\)）、
- 出力関数空間を \(H_{\text{out}}\) とする。

未知PDEが誘導する解作用素：
\[
G: H_{\text{in}} \to H_{\text{out}}
\]
データ：
\[
\{(a_i, u_i)\}_{i=1}^N,\qquad u_i = G(a_i).
\]

### 1.2 既知PDEの半群 \(T(t)=e^{tA}\)
既知線形PDEの生成子 \(A\) から強連続半群 \(T(t)\) を用いる。実装では **熱半群（拡散フィルタ）**を採用する：

- 1D/2D の周期領域 \([0,1]^d\) を想定（FFTで実装可能）
- \(A = \nu \Delta\) とし
\[
T(\tau)=e^{\tau \nu \Delta}.
\]
Fourier 空間での作用（このフィルタが「多スケール平滑化」を担う）：
\[
\widehat{T(\tau)a}(k) = e^{-\nu\tau (2\pi)^2 |k|^2}\,\hat a(k).
\]

### 1.3 ランダム特徴写像
確率変数 \(\omega=(g,\tau)\)：
- \(g\in H_{\text{in}}\) はランダムテスト関数（実装：離散グリッド上で i.i.d. Gaussian）
- \(\tau\ge 0\) はランダム時間（実装：uniform / log-uniform / exponential から選択）
- \(\sigma:\mathbb{R}\to\mathbb{R}\) は非多項式活性化（tanh/GELU/ReLU/sin を選択）

特徴：
\[
\phi_\omega(a) := \sigma(\langle g,\,T(\tau)a\rangle_{H_{\text{in}}}) \in \mathbb{R}.
\]

**重要（実装の高速化）：随伴（adjoint）トリック**
\[
\langle g, T(\tau)a\rangle = \langle T(\tau)^\* g, a\rangle.
\]
熱半群は自己共役なので \(T(\tau)^\*=T(\tau)\)。
よって各特徴に対し
\[
h_m := T(\tau_m)g_m
\]
を**事前計算**しておけば
\[
\phi_m(a) = \sigma(\langle h_m, a\rangle).
\]
これにより **学習時のFFTは特徴生成（初期化）時だけ**になり、各サンプルは内積（行列積）だけで特徴が計算できる。

### 1.4 出力の基底展開とモデル
出力基底 \(\{b_j\}_{j=1}^J\subset H_{\text{out}}\) を固定し、係数ベクトルで表す（離散では「grid基底」なら単にflatten）：
\[
\widehat G(a) = \sum_{j=1}^J\Big(\sum_{m=1}^M w_{jm}\,\phi_m(a)\Big)b_j.
\]

行列で：
- 特徴ベクトル \(\Phi(a)\in\mathbb{R}^M\)
- 重み \(W\in\mathbb{R}^{J\times M}\)
- 係数 \(\widehat y(a)=W\Phi(a)\in\mathbb{R}^J\)

### 1.5 RKHS／カーネルとしての見方（理論メモ；実装のスケーリングに関係）
この特徴写像は正定値核を誘導する：
\[
k(a,a') := \mathbb{E}_{\omega}\big[\phi_{\omega}(a)\,\phi_{\omega}(a')\big].
\]
有限個 \(M\) 個の特徴では
\[
k_M(a,a') := \frac{1}{M}\sum_{m=1}^M \phi_m(a)\phi_m(a')
= \big(\tfrac{1}{\sqrt{M}}\Phi(a)\big)^\top \big(\tfrac{1}{\sqrt{M}}\Phi(a')\big)
\]
で近似されるため、実装では `--feature-scale=inv_sqrt_m`（\(\Phi\leftarrow \Phi/\sqrt{M}\)）をデフォルト推奨とする。

出力がベクトル（係数） \(y\in\mathbb{R}^J\) の場合はベクトル値カーネル
\[
K(a,a') = k(a,a')\,I_J
\]
に対応し、本モデルは **(ベクトル値)カーネルリッジ回帰**の有限次元ランダム特徴近似になっている。

### 1.6 学習（多出力リッジ回帰）
\[
\min_{W\in\mathbb{R}^{J\times M}}
\sum_{i=1}^N \|W\Phi(a_i) - y_i\|_2^2 + \lambda\|W\|_F^2,
\]
ここで \(y_i\in\mathbb{R}^J\) は \(u_i\) の基底係数（gridならflatten）。

設計行列 \(\mathbf{\Phi}\in\mathbb{R}^{N\times M}\)（行が \(\Phi(a_i)^\top\)）、ターゲット \(\mathbf{Y}\in\mathbb{R}^{N\times J}\)（行が \(y_i^\top\)）として閉形式解：
\[
W^\top = (\mathbf{\Phi}^\top\mathbf{\Phi} + \lambda I)^{-1}\mathbf{\Phi}^\top\mathbf{Y}.
\]

実装は数値安定のため SPD 行列 \(A=\Phi^\top\Phi+\lambda I\) を **Cholesky** で解く。


---

## 2. 離散化とテンソル形状（Discrete spec / shapes）

### 2.1 1D（Burgers / `fourier_1d.py` dataset）
- データ：MATの `a`（入力）と `u`（出力）
- 読み込み形状（元）：`(nsamples, 2**13)` を `sub` で間引き → `S = 2**13 // sub`
- 本スクリプトの入力テンソル形状：
  - `x_train`: `(N, S, 1)`（チャネル1）
  - `y_train`: `(N, S)`（スカラー場）
- 特徴計算ではチャネルを squeeze して `(N, S)` にする。

### 2.2 2D（Darcy / `fourier_2d.py` dataset）
- データ：MATの `coeff`（入力）と `sol`（出力）
- downsample：`r` で `S = int(((grid_size-1)/r)+1)`（既存スクリプトと同じ）
- 形状：
  - `x_train`: `(N, S, S, 1)`
  - `y_train`: `(N, S, S)`
- 既存と同様に `UnitGaussianNormalizer` を使用：
  - `x_normalizer`: `x_train_raw -> x_train_norm`
  - `y_normalizer`: `y_train_raw -> y_train_norm`
- リッジ回帰のターゲットは **正規化後の `y_train_norm`**（推奨）。推論後に `y_normalizer.decode` して物理空間に戻し評価する。

### 2.3 内積（\(\langle h, a\rangle\)）の離散実装
連続 \(\int h(x)a(x)\,dx\) を安定に近似するため、実装では「平均」を使う：
- 1D：\(\langle h,a\rangle \approx \mathrm{mean}(h\odot a)\)
- 2D：\(\langle h,a\rangle \approx \mathrm{mean}(h\odot a)\)

（dx, dxdy の定数はモデルのスケールに吸収される。必要なら `--inner-product=mean|sum` で切替可能にしてもよい。）

### 2.4 特徴のスケーリング
ランダム特徴近似として典型的に
\[
\Phi(a) \leftarrow \frac{1}{\sqrt{M}}\Phi(a)
\]
のようにスケールさせる（オプション）。デフォルトは ON 推奨（数値安定）。


---

## 3. 実装する機能（What to implement）

### 3.1 追加ファイル（new files）
1. `pde_features.py`
   - 周波数グリッド生成（`torch.fft.fftfreq` / `rfftfreq`）
   - 熱半群フィルタ `apply_heat_semigroup_1d(x, tau, nu)` / `apply_heat_semigroup_2d(x, tau, nu)`
   - `PDERandomFeatureMap1D` / `PDERandomFeatureMap2D`（もしくは次元引数で統一）
     - 初期化で `tau_m` と `g_m` をサンプル
     - `h_m = T(tau_m) g_m` をベクトル化FFTで事前計算
     - `features(a_batch) -> (B, M)`
   - 活性化関数の factory `get_activation(name)`

2. `ridge.py`
   - `solve_ridge(Phi, Y, lam, jitter=1e-10, method="cholesky") -> W_T`
     - `Phi`: `(N,M)`, `Y`: `(N,J)`
     - `W_T`: `(M,J)` （= W^T）
   - 大きい `J` を想定して `Phi.T @ Y` を列方向に chunk で計算できる実装にする（メモリ節約）

3. `basis.py`
   - 最低限：
     - `GridBasis`：flatten/unflatten のみ（`J = S` or `S*S`）
   - 可能なら追加（推奨）：
     - `PODBasis`（PCA基底）
       - `fit(Y_train_flat, basis_dim, center=True) -> (U, mean)`
       - `encode(Y_flat) -> coeffs`
       - `decode(coeffs) -> Y_flat`
       - 実装は `torch.pca_lowrank`（あれば）か `torch.linalg.svd` fallback
       - 2Dで `J=S*S` が大きい時の圧縮に有効

4. `pde_rf_1d.py`
   - `fourier_1d.py` と同じデータモード（`single_split`/`separate_files`）と split 引数を提供
   - 学習は 1-shot（リッジ）で epochs は不要
   - `visualizations/pde_rf_1d/` に
     - 学習曲線（ここでは λ sweep しない限り1点なので省略可）
     - テスト誤差ヒストグラム
     - 1D予測プロット（数サンプル）
     を `viz_utils` で保存（png/pdf/svg）

5. `pde_rf_2d.py`
   - `fourier_2d.py` と同じデータモード（`single_split`/`separate_files`）
   - `UnitGaussianNormalizer` を同様に使用
   - `visualizations/pde_rf_2d/` に
     - テスト誤差ヒストグラム
     - 2D比較プロット（数サンプル）
     を保存

6. README更新（推奨）
   - 新スクリプトの実行例と主要CLI引数を追記

### 3.2 スモークテスト（必須）
データが無くても CI 的に確認できるよう、両スクリプトに
- `--smoke-test`
を追加し、MAT読み込みをせずに合成データでパイプラインを実行できるようにする。

推奨 smoke test:
- 入力 `a` は標準正規ランダム場
- 真の作用素 `G_true` は **熱半群**（同じ形でOK）：
  - 1D: `u = T(tau_true) a`
  - 2D: `u = T(tau_true) a`
- 小さなサイズ（例：1D `S=128`, 2D `S=32`）、小さな `N`（例：64/16）で動かす
- 期待：相対誤差が極端に大きくない（例：< 0.5 くらい）ことと、クラッシュしないこと


---

## 4. CLI仕様（Command-line interface）

### 4.1 `pde_rf_1d.py` の必須引数群
データ引数は `fourier_1d.py` と互換にする：
- `--data-mode {single_split,separate_files}`
- `--data-file`
- `--train-file --test-file`
- `--train-split --seed --shuffle`
- `--ntrain --ntest --sub`

PDE-RF固有：
- `--M`（特徴数; default: 2048）
- `--nu`（熱半群の拡散係数; default: 1.0）
- `--tau-dist {loguniform,uniform,exponential}`（default: loguniform）
- `--tau-min`（default: 1e-4）
- `--tau-max`（default: 1.0）
- `--tau-exp-rate`（exponential時のrate; default: 1.0）
- `--g-smooth-tau`（gに事前平滑化をかける熱半群時間; default: 0.0 = no smoothing）
- `--activation {tanh,gelu,relu,sin}`（default: tanh）
- `--feature-scale {none,inv_sqrt_m}`（default: inv_sqrt_m）
- `--ridge-lambda`（default: 1e-6）
- `--solve-device {auto,cpu,cuda}`（default: auto）
- `--dtype {float32,float64}`（default: float32）
- `--viz-dir`（default: visualizations/pde_rf_1d）
- `--num-viz`（default: 3）
- `--smoke-test`（bool）

任意（余裕があれば）：
- `--save-model PATH`：`torch.save` で (W_T, feature params, normalizers) を保存
- `--load-model PATH`：学習せず推論のみ

### 4.2 `pde_rf_2d.py` の必須引数群
データ引数は `fourier_2d.py` と互換：
- `--data-mode {single_split,separate_files}`
- `--data-file`
- `--train-file --test-file`
- `--ntrain --ntest`
- `--r --grid-size`

PDE-RF固有（1Dと同じ＋ basis）：
- `--M --nu --tau-dist --tau-min --tau-max --tau-exp-rate`
- `--g-smooth-tau --activation --feature-scale --ridge-lambda --solve-device --dtype`
- `--basis {grid,pod}`（default: grid）
- `--basis-dim`（pod時の次元; default: 256 など）
- `--pod-center/--no-pod-center`（default: center）
- `--viz-dir`（default: visualizations/pde_rf_2d）
- `--num-viz`（default: 3）
- `--smoke-test`

**注**：2Dは正規化の整合が重要。デフォルトでは
- 入力特徴は `x_train`（正規化後）で計算
- ターゲットは `y_train`（正規化後）で回帰
- 推論後 `y_normalizer.decode` して物理空間に戻して評価


---

## 5. 実装のアルゴリズム（Step-by-step; pseudo-code）

### 5.1 1D：学習
1. データロード（既存と同じ）
2. `x_train: (N,S,1) -> (N,S)` に整形
3. `feature_map = PDERandomFeatureMap1D(S, M, nu, tau_dist, ... , activation)`
4. バッチ処理で `Phi_train: (N,M)` を構築
5. `Y_train = y_train` を `grid` 基底で flatten（1Dなのでそのまま `(N,S)`）
6. `W_T = solve_ridge(Phi_train, Y_train, lam)`
7. テストも同様に `Phi_test` を作って
   - `Yhat_test = Phi_test @ W_T`（shape `(Ntest,J)`）
   - reshapeして `pred` を作成
8. `LpLoss`（既存 `utilities3.LpLoss`）や `viz_utils.rel_l2` で誤差評価
9. `viz_utils` でヒストグラム・サンプル可視化保存

### 5.2 2D：学習
1. データロード（既存と同じ）
2. `x_normalizer` / `y_normalizer` 作成（既存と同じ）
3. `x_train_norm, x_test_norm` を作成
4. `y_train_norm` を作成
5. `feature_map = PDERandomFeatureMap2D(S, M, ...)`
6. `Phi_train`/`Phi_test` をバッチで計算（入力は `x_*_norm.squeeze(-1)`）
7. `basis` を作る：
   - `grid`: `Y_train = y_train_norm.reshape(N, S*S)`
   - `pod` : `Y_train_flat` から `PODBasis.fit` → `coeff_train`
8. `W_T = solve_ridge(Phi_train, coeff_train, lam)`
9. 予測 `coeff_hat = Phi_test @ W_T`
10. `grid`: `yhat_norm = coeff_hat.reshape(Ntest,S,S)`
    `pod`: `yhat_norm_flat = basis.decode(coeff_hat)` → reshape
11. `yhat = y_normalizer.decode(yhat_norm)`
12. `LpLoss` を物理空間で計算（既存スクリプトと一致）
13. `viz_utils.plot_2d_comparison` で数サンプル可視化、誤差ヒストグラム保存


---

## 6. 数値安定・性能の注意点（Important implementation notes）

- `Phi.T @ Phi` は SPD になるように `lam>0` を必須にする（lam=0は不可）
- Cholesky が失敗する場合があるので
  - `jitter` を段階的に増やす（例：1e-10 → 1e-8 → 1e-6）
  - もしくは `torch.linalg.solve` fallback
- 大きい `J`（2D grid）では `B = Phi.T @ Y` のメモリが支配的になり得る：
  - `Y` を列方向 chunk（例：1024列ずつ）で `B` を作る実装にする
- `solve_device`：
  - 行列分解はCPUのほうが安定な場合がある。`--solve-device cpu` を用意
- dtype：
  - デフォルト float32
  - 不安定な場合は float64 をオプションにする
- Feature precompute：
  - `h_m` 計算は `torch.fft.rfft/rfft2` を M バッチとしてまとめて実行し高速化する
- 再現性：
  - `--seed` で `torch.manual_seed` と `np.random.seed` を設定
  - `tau` と `g` のサンプルも同じ seed に従う


---

## 7. 完了条件（Acceptance criteria / Definition of Done）

最低限、以下を満たすこと：

1. `python pde_rf_1d.py --help` が動き、CLIが説明的
2. `python pde_rf_2d.py --help` が動き、CLIが説明的
3. `python pde_rf_1d.py --smoke-test` がクラッシュせず最後まで走る
4. `python pde_rf_2d.py --smoke-test` がクラッシュせず最後まで走る
5. 実データがある環境では、既存データ引数で
   - `pde_rf_1d.py` が Burgers データを読み込み評価まで進む
   - `pde_rf_2d.py` が Darcy データを読み込み評価まで進む
6. 予測可視化が `visualizations/pde_rf_1d/` と `visualizations/pde_rf_2d/` に png/pdf/svg で保存される

（精度は研究段階なので閾値は固定しないが、少なくとも smoke test では極端に悪化しないこと。）


---

## 8. 追加でできると良い（Nice-to-have; optional）
- `--save-model` / `--load-model` の実装（再学習なし推論）
- `--basis=pod` を 1Dにも適用（任意）
- README に「PDE-RF vs FNO」の比較実行例を追記
- “既知PDEを変える”実験のため、`--semigroup {heat,helmholtz,...}` の拡張余地をコメントで残す
# AGENT.md — PDE-RF: Add Known-PDE Operator Families (Candidates A–E)

## 0. Goal
This repo already contains a working **PDE-induced Random Feature Operator Learning (PDE-RF)** baseline:

- Random features are built from a **known linear PDE operator** \(T_\theta\) (semigroup / linear propagator / resolvent).
- The unknown PDE solution operator \(G\) is regressed via **multi-output ridge regression**.

However, the current implementation uses **only the heat semigroup** (diffusion), which can underperform for problems where **transport/phase** (e.g., Burgers) or **elliptic inverse structure** (e.g., Darcy) matters.

### Your task
Extend the PDE feature family to implement **all candidates A–E** as selectable operators:

- **A (1D)**: Advection (transport) operator
- **B (1D)**: Convection–Diffusion operator
- **C (1D)**: Wave / Damped-wave filter operator (linear oscillatory)
- **D (2D)**: Helmholtz / Screened-Poisson resolvent operator
- **E (2D)**: Anisotropic diffusion operator

You must integrate them into the existing PDE-RF scripts:

- `pde_rf_1d.py` (Burgers-style datasets) — add operators A/B/C (heat stays available)
- `pde_rf_2d.py` (Darcy-style datasets) — add operators D/E (heat stays available)

All operators must support **random parameter sampling per-feature**, precomputation of \(h_m\), and fast feature computation on batches.

---

## 1. Mathematical specification (must match implementation)

### 1.1 Operator learning setup
We learn an operator
\[
G: H_{\text{in}} \to H_{\text{out}}
\]
from data \(\{(a_i,u_i)\}_{i=1}^N\), \(u_i = G(a_i)\).

### 1.2 Random features using a known PDE-induced linear operator
For each feature \(m=1,\dots,M\), sample random parameters \(\omega_m\) (includes a random test function \(g_m\) and operator parameters like \(\tau\), \(c\), \(\alpha\), etc.).

Define features
\[
\phi_m(a) := \sigma\big(\langle g_m,\; T_{\omega_m}(a)\rangle\big) \in \mathbb{R}
\]
where \(\sigma\) is a non-polynomial activation (tanh/gelu/relu/sin).

#### Adjoint trick (critical for speed)
We must avoid applying \(T\) to every sample \(a\). Use
\[
\langle g, T(a) \rangle = \langle T^*(g), a \rangle
\]
and precompute
\[
h_m := T_{\omega_m}^*(g_m).
\]
Then the feature becomes
\[
\boxed{\phi_m(a) = \sigma(\langle h_m, a\rangle).}
\]

This reduces runtime feature extraction to **inner products only**.

### 1.3 Model and learning (ridge regression)
Let \(\Phi(a)\in\mathbb{R}^M\) be stacked features.

Output is represented in a finite basis (default: grid basis = flatten):
- 1D: \(J=S\)
- 2D: \(J=S^2\) (or POD basis \(J=\text{basis_dim}\))

We learn weights \(W\in\mathbb{R}^{J\times M}\) via multi-output ridge regression:
\[
\min_W \sum_{i=1}^N \|W\Phi(a_i)-y_i\|_2^2 + \lambda\|W\|_F^2
\]
with closed-form solution
\[
W^\top = (\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top Y.
\]
Implementation should use **Cholesky** for SPD stability.

---

## 2. Discrete setting (what tensors mean)

### 2.1 Discrete inner product
Use the repo’s current convention:
- `inner_product = "mean"`: \(\langle h,a\rangle \approx \mathrm{mean}(h\odot a)\)
- `inner_product = "sum"`: \(\langle h,a\rangle \approx \mathrm{sum}(h\odot a)\)

(Keeping this configurable is required for backward compatibility.)

### 2.2 FFT grid conventions
All operators A–E will be implemented on periodic grids using FFTs:
- 1D: `torch.fft.rfft / irfft`
- 2D: `torch.fft.rfft2 / irfft2`
Frequencies use:
- 1D: `torch.fft.rfftfreq(S, d=1.0/S)`
- 2D: `kx = fftfreq(S)`, `ky = rfftfreq(S)` to match `rfft2` shapes.

---

## 3. Candidate operators A–E (exact Fourier multipliers)

Below, \(k\) denotes integer frequency index (as returned by `fftfreq/rfftfreq`), and **physical wavenumber** is \(2\pi k\). Always use \( (2\pi)^2 \) factors exactly as written.

### 3.1 (Existing) Heat semigroup (keep)
PDE: \(u_t = \nu \Delta u\)

Fourier multiplier:
\[
\widehat{T_{\text{heat}}(\tau)a}(k) = \exp\!\left(-\nu\tau(2\pi)^2|k|^2\right)\hat a(k).
\]
Self-adjoint: \(T^*=T\).

### 3.2 Candidate A — 1D Advection (transport)
PDE: \(u_t + c\,u_x = 0\)

Semigroup = shift:
\[
(T_{\text{adv}}(\tau;c)a)(x) = a(x-c\tau).
\]

Fourier multiplier (periodic):
\[
\widehat{T_{\text{adv}}(\tau;c)a}(k)=\exp\!\left(-i(2\pi)(c\tau)k\right)\hat a(k).
\]

Adjoint in \(L^2\) (periodic) flips velocity:
\[
T_{\text{adv}}(\tau;c)^* = T_{\text{adv}}(\tau;-c)
\]
because the multiplier conjugates.

**Implementation rule for features:** precompute \(h = T^* g\) by applying advection with `c_adj = -c`.

### 3.3 Candidate B — 1D Convection–Diffusion
PDE: \(u_t + c\,u_x = \nu u_{xx}\)

Fourier multiplier:
\[
\widehat{T_{\text{cd}}(\tau;\nu,c)a}(k)
=
\exp\!\left(-\nu\tau(2\pi)^2k^2\right)\cdot
\exp\!\left(-i(2\pi)(c\tau)k\right)\hat a(k).
\]

Adjoint flips the advection sign (diffusion is self-adjoint):
\[
T_{\text{cd}}(\tau;\nu,c)^* = T_{\text{cd}}(\tau;\nu,-c).
\]

**Implementation rule for features:** precompute \(h=T^*g\) using `c_adj=-c`.

### 3.4 Candidate C — 1D Wave / Damped Wave filter
We use the *mapping from initial displacement \(a(x)\) to displacement at time \(\tau\)* with zero initial velocity.

Option 1 (undamped wave filter):
\[
u_{tt} = c_w^2 u_{xx},\quad u(0)=a,\ u_t(0)=0
\]
\[
\widehat{T_{\text{wave}}(\tau;c_w)a}(k) = \cos\!\big((2\pi)c_w\tau |k|\big)\,\hat a(k).
\]

Option 2 (damped wave filter, recommended for stability):
\[
u_{tt} + \gamma u_t = c_w^2 u_{xx},\ u(0)=a,\ u_t(0)=0
\]
Use the simplified multiplier:
\[
\widehat{T_{\text{dwave}}(\tau;c_w,\gamma)a}(k)
=
e^{-\gamma\tau/2}\cos\!\big((2\pi)c_w\tau |k|\big)\,\hat a(k).
\]
(This is real, even in \(k\), and works well as a linear oscillatory filter.)

Self-adjoint (real multiplier): \(T^*=T\).

**Implementation rule for features:** \(h = T g\).

### 3.5 Candidate D — 2D Helmholtz / Screened Poisson resolvent
Solve for \(u\):
\[
(\alpha I - \nu\Delta)u = a,\qquad \alpha>0
\]
so the operator is
\[
T_{\text{helm}}(\alpha)a = (\alpha I - \nu\Delta)^{-1}a.
\]

Fourier multiplier:
\[
\widehat{T_{\text{helm}}(\alpha)a}(k)
=
\frac{1}{\alpha + \nu(2\pi)^2|k|^2}\,\hat a(k).
\]
Self-adjoint: \(T^*=T\).

**Notes:**
- This is not a time semigroup; treat `alpha` as the per-feature parameter (sampled similarly to `tau`).
- This operator decays high frequencies polynomially, often preserving more detail than heat.

### 3.6 Candidate E — 2D Anisotropic diffusion
PDE:
\[
u_t = \nabla\cdot(D\nabla u)
\]
with \(D\in\mathbb{R}^{2\times2}\) symmetric positive definite (SPD).

Fourier multiplier:
\[
\widehat{T_{\text{aniso}}(\tau;D)a}(k)
=
\exp\!\left(-(2\pi)^2\tau\,(k^\top D k)\right)\hat a(k),
\]
where \(k=(k_x,k_y)\) matches the FFT frequency grids.

Self-adjoint for SPD \(D\): \(T^*=T\).

**Parameterization for random SPD \(D\) (must implement):**
Sample per-feature:
- rotation angle \(\theta\sim \mathrm{Uniform}(0,\pi)\)
- eigenvalues \(d_1,d_2>0\) (uniform or loguniform in \([d_{\min}, d_{\max}]\))

Let \(R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{bmatrix}\),
\[
D = R\operatorname{diag}(d_1,d_2)R^\top.
\]
Store components \(D_{11},D_{12},D_{22}\) per feature.

---

## 4. What to change in code (exact integration plan)

### 4.1 `pde_features.py` — extend to multi-operator feature maps
You must implement:

#### (A) Operator application functions (vectorized over batch/feature dims)
1D:
- `apply_advection_1d(x, tau, c)`
- `apply_convection_diffusion_1d(x, tau, nu, c)`
- `apply_wave_1d(x, tau, c_wave, gamma)`  (gamma may be 0)
(Heat already exists.)

2D:
- `apply_helmholtz_2d(x, alpha, nu)`
- `apply_anisotropic_diffusion_2d(x, tau, d11, d12, d22)` (or accept `D` packed)
(Heat already exists.)

All functions must support:
- `x` with leading batch dims and last dims = spatial dims
- `tau/alpha/c/...` as either python floats or `torch.Tensor` with shape broadcastable to leading dims (esp. per-feature tensors of shape `(M,)`).
- correct `device`/`dtype`
- output real-valued tensor of same real dtype as input.

#### (B) Parameter samplers
Keep existing `_sample_tau`. Add:
- `_sample_c(m, dist, c_min, c_max, c_std, c_fixed, ...) -> Tensor`
- `_sample_alpha(m, dist, alpha_min, alpha_max, ...) -> Tensor`
- `_sample_spd_2x2(m, eig_dist, eig_min, eig_max, theta_dist, ...) -> (d11,d12,d22)`

#### (C) Update feature map classes to accept `operator` and operator-specific params
Update `PDERandomFeatureMap1D` and `PDERandomFeatureMap2D` signatures to include:
- `operator: str` with choices:
  - 1D: `heat`, `advection`, `convdiff`, `wave`
  - 2D: `heat`, `helmholtz`, `aniso`
(Keep `heat` default to maintain backward compatibility.)

Feature map initialization must:
1. sample `tau` (for time-like operators: heat/advection/convdiff/wave/aniso)
2. sample operator parameters (e.g., c for advection; alpha for helmholtz; D for aniso; c_wave/gamma for wave)
3. sample random test functions `g ~ N(0,I)`
4. optional: `g_smooth_tau` pre-smoothing (keep existing behavior; use heat smoothing as pre-filter)
5. compute `h = T^* g` using the correct adjoint rule:
   - heat: `h = apply_heat_semigroup(g, tau, nu)`
   - advection: `h = apply_advection_1d(g, tau, c=-c)`
   - convdiff: `h = apply_convection_diffusion_1d(g, tau, nu, c=-c)`
   - wave: `h = apply_wave_1d(g, tau, c_wave, gamma)` (self-adjoint)
   - helmholtz: `h = apply_helmholtz_2d(g, alpha, nu)` (self-adjoint)
   - aniso: `h = apply_anisotropic_diffusion_2d(g, tau, d11, d12, d22)` (self-adjoint)
6. store buffers for all parameters (`tau`, `c`, `alpha`, `d11/d12/d22`, etc.) and `h` (and `h_flat` in 2D).

Feature extraction remains:
- 1D: `inner = a_batch @ h.T`
- 2D: `inner = a_flat @ h_flat.T`

Keep:
- `inner_product` scaling
- `feature_scale` option (`inv_sqrt_m`)

### 4.2 `pde_rf_1d.py` — add CLI flags and wire operator selection
Add new CLI args:
- `--operator {heat,advection,convdiff,wave}` (default: heat)

Advection / Convection–Diffusion parameters:
- `--c-dist {uniform,normal,fixed}` (default: uniform)
- `--c-max` (default: 1.0)  (uniform uses [-c_max, c_max])
- `--c-std` (default: 1.0)  (normal uses N(0, c_std^2))
- `--c-fixed` (default: 1.0) (fixed uses this constant)

Wave parameters:
- `--wave-c-dist {uniform,loguniform,fixed}` (default: uniform)
- `--wave-c-min` (default: 0.1)
- `--wave-c-max` (default: 2.0)
- `--wave-c-fixed` (default: 1.0)
- `--wave-gamma` (default: 0.0)  # damping coefficient (>=0)

Keep existing `--nu`, `--tau-*`, `--activation`, etc. `--nu` is used for heat and convdiff.

Update feature map construction to pass all operator params.

Update `--smoke-test` to work with *the selected operator*:
- If `--smoke-test` is set, generate synthetic data where the true mapping is exactly the same operator family with fixed “true” parameters:
  - heat: `u = apply_heat_semigroup_1d(a, tau_true, nu_true)`
  - advection: `u = apply_advection_1d(a, tau_true, c_true)`
  - convdiff: `u = apply_convection_diffusion_1d(a, tau_true, nu_true, c_true)`
  - wave: `u = apply_wave_1d(a, tau_true, c_wave_true, gamma_true)`
This ensures each operator path is exercised end-to-end.

### 4.3 `pde_rf_2d.py` — add CLI flags and wire operator selection
Add new CLI arg:
- `--operator {heat,helmholtz,aniso}` (default: heat)

Helmholtz parameters:
- `--alpha-dist {uniform,loguniform,fixed}` (default: loguniform)
- `--alpha-min` (default: 1e-2)
- `--alpha-max` (default: 10.0)
- `--alpha-fixed` (default: 1.0)

Anisotropic diffusion parameters:
- `--aniso-eig-dist {uniform,loguniform}` (default: loguniform)
- `--aniso-eig-min` (default: 1e-3)
- `--aniso-eig-max` (default: 1.0)
- `--aniso-theta-dist {uniform}` (default: uniform)  # theta ~ U(0, pi)

`--nu` is used for heat and helmholtz.

Update feature map construction to pass these operator params.

Update `--smoke-test` to work with the selected operator:
- heat: `u = apply_heat_semigroup_2d(a, tau_true, nu_true)`
- helmholtz: `u = apply_helmholtz_2d(a, alpha_true, nu_true)`
- aniso: `u = apply_anisotropic_diffusion_2d(a, tau_true, D_true)`

### 4.4 README (optional but recommended)
Add a short section showing how to run each operator:
- 1D: `--operator advection|convdiff|wave`
- 2D: `--operator helmholtz|aniso`

---

## 5. Numerical stability & correctness checks (must do)

### 5.1 Correctness checks in code
Add lightweight asserts / shape checks:
- outputs are real dtype (`torch.is_floating_point`)
- shapes match input shapes
- broadcasting works for per-feature params (tau shape `(M,)`)

### 5.2 Solve stability
Keep ridge solve as-is (Cholesky with jitter fallback). Do not remove chunked `Phi.T@Y`.

### 5.3 Compatibility
Try to avoid Python 3.10-only syntax in NEW code (e.g., avoid `Tensor | float` in new functions).
If you refactor existing annotations, keep behavior identical.

---

## 6. Acceptance criteria (Definition of Done)

### 6.1 CLI help works
- `python pde_rf_1d.py --help` shows new flags.
- `python pde_rf_2d.py --help` shows new flags.

### 6.2 Smoke tests: **all operators must run**
1D:
- `python pde_rf_1d.py --smoke-test --operator heat`
- `python pde_rf_1d.py --smoke-test --operator advection`
- `python pde_rf_1d.py --smoke-test --operator convdiff`
- `python pde_rf_1d.py --smoke-test --operator wave`

2D:
- `python pde_rf_2d.py --smoke-test --operator heat`
- `python pde_rf_2d.py --smoke-test --operator helmholtz`
- `python pde_rf_2d.py --smoke-test --operator aniso`

Each run must:
- complete without crashing
- save visualizations into `visualizations/pde_rf_1d/` or `visualizations/pde_rf_2d/`

### 6.3 Syntax check
- `python -m py_compile pde_features.py ridge.py basis.py pde_rf_1d.py pde_rf_2d.py` passes.

---

## 7. Notes: why these operators
- A/B/C add **phase/transport/oscillation** structure (critical for Burgers-like dynamics).
- D matches elliptic inverse structure better than heat (often improves Darcy-like tasks).
- E adds **directional multi-scale smoothing** (captures anisotropic patterns).

Implement them exactly as specified, keeping existing heat functionality intact.
