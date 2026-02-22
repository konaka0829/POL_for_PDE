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
