# AGENT.md — PDE Reservoir Operator Learning (Target: 1D Burgers) + ELM + Ridge (Backprop-free)

## 0. このリポジトリの前提（現状）
- この repo は Fourier Neural Operator (FNO) 実装で、`fourier_1d.py` が Burgers データセット（`.mat`）を読み、
  入力 `a`（初期条件）→ 出力 `u`（時刻1の解）を学習する構成。
- データ読み込み・分割の CLI は `cli_utils.py` を使用。
- 可視化ユーティリティは `viz_utils.py`（PNG/PDF/SVG の3形式保存）。

**この追加機能は、既存のFNO/lowrankスクリプトを壊さずに「新規スクリプト＋共通モジュール追加」で実装する。**

---

## 1. 実装したい新手法の要約（本タスクの目的）
### 1.1 ターゲット（学習対象）
- ターゲット PDE は **Burgers データセット**（`fourier_1d.py` が使っているもの）と同じ。
- 学習したい作用素：  
  **F : a(x) ↦ u(T=1, x)**  
  （データ `.mat` では `a` が入力、`u` が出力）

### 1.2 モデル（学習器）
以下の backprop-free パイプラインを実装する：

1) **入力** `a` を **リザーバPDEの初期条件**として入れる（エンコーダは基本 `E(a)=scale*a + shift`）
2) **固定の既知PDE（リザーバPDE）** を時間発展させる：  
   `z_t = R(z)` （境界は周期を仮定）
3) 観測 `Obs(z(t_k))` を K 個の時刻で取り、特徴ベクトルを作る：  
   `Φ(a) = concat_k Obs(z(t_k))`
4) **拡張(3)**：固定ランダム写像（ELM）を入れる：  
   `h(a) = σ( Φ(a) @ A^T + c )`  
   - A, c は固定（乱数）
   - σ は tanh / ReLU など
5) 最後の線形 readout を **リッジ回帰**（閉形式 or XtX/XtY 蓄積）で解く：  
   `ŷ = [h, 1] W_out`

### 1.3 重要要件
- **学習するのは W_out（とバイアス）だけ**。リザーバPDE・Obs・ELMは固定。
- 3種類のリザーバPDEを **CLIで切替可能**にする：
  1. `reaction_diffusion`（1D反応拡散：例 Allen–Cahn）
  2. `ks`（1D Kuramoto–Sivashinsky）
  3. `burgers`（別パラメータの Burgers：粘性などを変える）
- 既存のデータ読み込み方式（`--data-mode single_split / separate_files`）に対応。

---

## 2. 新規追加するファイル構成（提案）
repo ルート（README と同階層）に以下を追加する：

- `reservoir_burgers_1d.py`  
  - 本手法のメイン実行スクリプト（Burgers ターゲット専用）
  - CLI で reservoir/ELM/ridge/feature の設定ができる

- `pol/` ディレクトリ（新規、パッケージ扱い）
  - `pol/__init__.py`
  - `pol/reservoir_1d.py`  
    - 3種のリザーバPDEソルバ実装（PyTorch）
  - `pol/features_1d.py`  
    - multi-time feature 作成（Φ）、観測（Obs）、正規化など
  - `pol/elm.py`  
    - 固定ランダム層（A,c, activation）
  - `pol/ridge.py`  
    - リッジ回帰（XtX/XtY 蓄積→Cholesky solve）

- `tests/test_reservoir_smoke.py`（任意だが推奨）
  - 外部データ無しで shape/NaN を確認する簡易テスト

- README 更新（推奨）
  - 新手法の使い方（実行例・主要引数）

※この repo は「スクリプト中心」なので、モジュール分割は最小限でOK。ただし保守性のため `pol/` 分離を推奨。

---

## 3. 数学仕様（Codex が迷わないための明文化）

### 3.1 共通：領域・離散化・境界
- 空間領域：`x ∈ [0,1]`（`fourier_1d.py` の grid と整合）
- 離散点数：`s = 2**13 // sub`（現行と同様）
- 境界条件：**周期境界**（spectral/FFT を使う前提）
- 1サンプルの状態：`z ∈ R^s`

### 3.2 1Dスペクトル微分（rFFT）
- `dx = L / s`（L=1）
- `k = 2π * rfftfreq(s, d=dx)`（shape: `s//2 + 1`）
- `FFT(u) = û`
- `u_x = irfft( (i k) û )`
- `u_xx = irfft( -(k^2) û )`
- `u_xxxx = irfft( (k^4) û )`

### 3.3 リザーバPDE（3種類）と時間発展
全て **バッチ**で計算できる形（`(B, s)`）で実装する。

#### (A) reaction_diffusion（例：Allen–Cahn 型）
- 方程式（推奨）：
  - `z_t = ν z_xx + α z - β z^3`
- 時間積分（推奨：semi-implicit Euler）
  - 拡散項を implicit、反応項を explicit：
    - `N(z) = α z - β z^3`
    - `z_hat_next = (z_hat + dt * FFT(N(z))) / (1 + dt * ν k^2)`

CLI パラメータ例：
- `--rd-nu`（例 1e-3）
- `--rd-alpha`（例 1.0）
- `--rd-beta`（例 1.0）

#### (B) ks（Kuramoto–Sivashinsky）
- 標準形（周期）：
  - `z_t + z z_x + z_xx + z_xxxx = 0`
  - `=> z_t = - z z_x - z_xx - z_xxxx`
- ここで
  - `N(z) = - z z_x`
  - 線形項のフーリエ表現：  
    `L_hat = (k^2 - k^4)` （※上式に従う）
- semi-implicit Euler：
  - `z_hat_next = (z_hat + dt * FFT(N(z))) / (1 - dt * L_hat)`

CLI パラメータ例：
- `--ks-dt`（小さめ推奨、例 1e-3〜5e-4）
- `--Tr`（例 0.5〜1.0）
- （必要なら）`--ks-dealias`（2/3 ルールで高周波カット）

#### (C) burgers（別パラメータBurgers）
- 方程式：
  - `z_t + z z_x = ν z_xx`
  - `=> z_t = - z z_x + ν z_xx`
- `N(z) = - z z_x`
- semi-implicit Euler：
  - `z_hat_next = (z_hat + dt * FFT(N(z))) / (1 + dt * ν k^2)`

CLI パラメータ例：
- `--res-burgers-nu`（例：ターゲットと違う値にする。例 0.05 など）
- `--dt`（CFL的に小さめ推奨：s=1024なら 1e-3 程度）

### 3.4 特徴抽出：multi-time + 観測
- 観測時刻 `t_k` を K 個用意：
  - CLI で
    - `--K` と `--Tr` から等間隔に作る（推奨）
    - または `--feature-times "0.1,0.2,0.5"` のように直接指定
- 観測 `Obs` は最低限 2 種類：
  1) `full`：状態全点（`Obs(z)=z`）
  2) `points`：J点の点観測（固定インデックス）
     - `--J` と `--sensor-mode equispaced|random`
     - random の場合 `--sensor-seed` で固定

- 特徴 `Φ(a)` の shape：
  - `full` の場合：`M = K * s`
  - `points` の場合：`M = K * J`

### 3.5 拡張(3) ELM 固定ランダム非線形
- `Φ ∈ R^{M}` に対し
  - `A ∈ R^{H×M}`, `c ∈ R^{H}` を固定乱数で生成
  - `h = σ( Φ A^T + c ) ∈ R^{H}`
- activation `σ` は `tanh` / `relu` を選択可能に。

CLI パラメータ例：
- `--elm-h`（H、例 2048）
- `--elm-activation tanh|relu`
- `--elm-seed`
- `--elm-weight-scale`（Aの標準偏差の係数。例 1/sqrt(M) を基準に調整）
- `--elm-bias-scale`

※ELMを無効化して `h=Φ` とするモードもあると便利：
- `--use-elm 0/1`

### 3.6 リッジ回帰（backprop-free 学習）
- 学習データ：
  - `X = h(a_i)` を並べて `X ∈ R^{N×H}`
  - バイアス込み：`X̃ = [X, 1] ∈ R^{N×(H+1)}`
  - `Y ∈ R^{N×s}`（Burgers は出力が全格子）
- リッジ解：
  - `W = (X̃^T X̃ + λ I)^(-1) X̃^T Y`
- 実装は **逆行列を作らず**、Cholesky or solve を使う。

大規模に備えて、`X̃` を全部保存せずに
- `G = X̃^T X̃`（(H+1)×(H+1)）
- `S = X̃^T Y`（(H+1)×s）
をミニバッチで蓄積して最後に解く方式を実装する（推奨）。

CLI：
- `--ridge-lambda`（例 1e-6〜1e-2）
- optional: `--ridge-dtype float64`（安定化）

---

## 4. 実装要件（具体）
### 4.1 `reservoir_burgers_1d.py` の要件
- `fourier_1d.py` と同様のデータ読み込み CLI に対応する：
  - `--data-mode`, `--data-file`, `--train-file`, `--test-file`, `--train-split`, `--seed`, `--shuffle`
  - `--ntrain`, `--ntest`, `--sub`, `--batch-size`
- 追加 CLI（必須）：
  - `--reservoir reaction_diffusion|ks|burgers`
  - `--Tr`, `--dt`
  - `--K` または `--feature-times`
  - `--obs full|points`
  - `--J`, `--sensor-mode`, `--sensor-seed`（points 用）
  - `--input-scale`, `--input-shift`（E(a) 用）
  - ELM: `--use-elm`, `--elm-h`, `--elm-activation`, `--elm-seed`, `--elm-weight-scale`, `--elm-bias-scale`
  - Ridge: `--ridge-lambda`, `--ridge-dtype float32|float64`
  - `--device auto|cpu|cuda`
  - `--out-dir`（結果・可視化出力先）
  - `--save-model`（torch.save で辞書保存：W_out, A, c, sensor_idx, config など）

- 出力（標準出力）：
  - train relL2 / test relL2（少なくとも test）
  - 実行時間（任意）
- 可視化（`viz_utils.py` を使用）：
  - テストの per-sample relL2 ヒストグラム
  - 代表サンプルの 1D プロット（GT vs pred + input）
  - すべて PNG/PDF/SVG で保存

※本手法は epoch 学習がないので learning curve は不要（代わりに λ や reservoir 切替の結果をログ保存するのが良い）

### 4.2 モジュール側の要件
- `pol/reservoir_1d.py`
  - 3種類の reservoir の共通インターフェース：
    - `simulate(z0: Tensor[B,s], dt: float, Tr: float, obs_steps: list[int]) -> list[Tensor[B,s]]`
  - `torch.no_grad()` 前提で高速に
  - wave number `k` は (s, device, dtype) ごとに再計算しないようキャッシュする（推奨）
- `pol/features_1d.py`
  - 1) `obs_steps` の生成（K/Tr/dt or feature-times）
  - 2) `Obs` 実装（full/points）
  - 3) `Φ` の組み立て（concat）
  - 4) （任意）`Φ` の標準化（train mean/std で固定）
- `pol/elm.py`
  - fixed random matrix A, bias c
  - activation
  - forward: `phi -> h`
- `pol/ridge.py`
  - `fit_ridge_streaming(dataloader, feature_fn, lambda, dtype) -> W`
  - `predict(feature_fn, W)`

---

## 5. 受け入れ条件（Acceptance Criteria）
Codex が実装した後、最低限これが満たされること：

1) **既存スクリプトが壊れていない**（`fourier_1d.py` などの動作はそのまま）
2) `python -m py_compile reservoir_burgers_1d.py pol/*.py` が通る
3) `python reservoir_burgers_1d.py --dry-run`（外部データ無しのランダム入力で shape/NaN チェック）が通る  
   - dry-run は `--ntrain 8 --ntest 4 --sub 256` 相当の内部乱数データでも良い
4) 実データがある環境では、例のコマンドで最後まで完走し、test relL2 を表示・図を保存できる：

例：
```bash
python reservoir_burgers_1d.py \
  --data-mode single_split --data-file data/burgers_data_R10.mat \
  --ntrain 1000 --ntest 100 --sub 8 --batch-size 20 \
  --reservoir reaction_diffusion --Tr 1.0 --dt 0.01 --K 5 --obs full \
  --use-elm 1 --elm-h 2048 --elm-activation tanh --elm-seed 0 \
  --ridge-lambda 1e-4 --ridge-dtype float64 \
  --out-dir visualizations/reservoir_burgers_rd
```

# AGENT.md — 既存PDEリザーバ実装に (1)特徴標準化 (2)Fourier/Projection観測 (3)関数値RFM(m×m) を追加する

## 0. このリポジトリの現状（Codexが迷わないための前提）
- 既に backprop-free の **PDEリザーバ + 観測(Obs) + (任意)ELM + リッジ回帰** が実装されている。
- メインスクリプト: `reservoir_burgers_1d.py`
- 共通モジュール: `pol/` 配下
  - `pol/reservoir_1d.py` : 1D periodic reservoir PDE solver（reaction_diffusion / ks / burgers）
  - `pol/features_1d.py` : 観測時刻グリッド、センサ、観測収集、flatten
  - `pol/elm.py` : 固定ランダム ELM
  - `pol/ridge.py` : streaming ridge（Gram/Cross蓄積→Cholesky solve）
- `reservoir_burgers_1d.py` は `--dry-run` で外部データなしでも動く。

本タスクは、**既存のFNO系スクリプトやデータ読み込み方式を壊さず**、`reservoir_burgers_1d.py` と `pol/` を中心に「3つの改善」を追加する。

---

## 1. 目的（今回追加したい 3 点）
### 1) --input-scale/shift と「特徴標準化（mean/std）」を実装に組み込む
- `--input-scale/shift` は既に存在し、`phi_fn` 内で初期条件 `z0 = scale*x + shift` を作っている。
- 追加したいのは **最終特徴（ELM後の h）** を学習データ統計（mean/std）で標準化してからリッジを解くこと。

重要: 特徴標準化を “2パスで特徴抽出し直す” と PDEシミュレーションが倍になって重い。
→ **1パスで蓄積した Gram/Cross から mean/std を復元し、解析的に標準化座標へ変換して解く** 実装にする。

### 2) 観測 Obs を Fourier / projection に拡張（点サンプル以外の線形汎関数）
- 既存の `obs` は `full|points` のみ。
- 追加する `obs`:
  - `fourier`: 低周波 rFFT モードを取る（Re/Im を連結して実数特徴にする）
  - `proj`: ランダムテスト関数（行列）への射影（線形汎関数）
- いずれも backprop-free を壊さない（固定線形観測）。

### 3) ルート3：PDFのRFMそのもの（関数値ランダム特徴の線形結合）として組み直すスクリプトを追加
- 出力が関数（格子関数）で、特徴も関数（格子関数）:
  - `φ_j(a) ∈ R^s` を m 個用意
  - 予測: `ŷ(a) = Σ_j α_j φ_j(a)`
  - 学習は `m×m` の線形方程式を解く（出力次元 s に依存しない）
- これを `rfm_burgers_1d.py` として新規追加する（既存 `reservoir_burgers_1d.py` は保持）。

---

## 2. 数学仕様（実装を迷わないための完全定義）

### 2.1 元の実装：PDEリザーバ + 観測 + (任意)ELM + リッジ
入力 `a ∈ R^s`（Burgersデータの `a`）、出力 `y ∈ R^s`（`u`）。

(1) エンコード（現状のまま）
- `z0(a) = α a + β`
  - α = `--input-scale`
  - β = `--input-shift`（スカラー、全点にブロードキャスト）

(2) リザーバPDE（既存実装）
`pol/reservoir_1d.py` の `Reservoir1DSolver.simulate(z0, dt, Tr, obs_steps)` が返す
`states = [z(t_k)]_{k=1..K}`（shape 各 `(B,s)`）を使う。

(3) 観測（既存 + 拡張）
各時刻状態 `z(t_k)` に線形観測 `Obs` を適用:
- full: `Obs(z)=z ∈ R^s`
- points: `Obs(z) = (z[p1],...,z[pJ]) ∈ R^J`
- fourier: `Obs(z) = [Re(ẑ_0..ẑ_{J-1}), Im(ẑ_0..ẑ_{J-1})] ∈ R^{2J}`
- proj: `Obs(z) = Ψ z ∈ R^J`（Ψ ∈ R^{J×s} 固定）

(4) multi-time concat
`Φ(a) = concat_k Obs(z(t_k)) ∈ R^M`

(5) 任意の固定ランダム ELM
`h(a) = σ(A Φ(a) + c) ∈ R^H`（ELM無効なら h=Φ）

(6) リッジ回帰
バイアス付き `x̃=[h,1]` として
`min_W Σ_i || x̃_i^T W - y_i ||^2 + λ ||W||^2`（バイアス行は正則化しない）
を streaming で解く（既存 `fit_ridge_streaming`）。

### 2.2 追加：特徴標準化（mean/std）込みリッジ
最終特徴 `h_i ∈ R^H` の平均と標準偏差（要素ごと）
- `mean = (1/N) Σ h_i`
- `std = sqrt( (1/N) Σ h_i^2 - mean^2 )`
- `h_std = (h - mean)/(std+eps)`

目的は `h_std` を使ってリッジを解くことだが、
PDE計算を二度回さないため、**1パスで蓄積した Gram/Cross から標準化座標の Gram/Cross を構成して解く**。

要求: 実装は `pol/ridge.py` に `fit_ridge_streaming_standardized(...)` を追加し、
- 返す `W` は **raw特徴 h に直接作用**する重みで `predict_linear(h, W)` がそのまま使えること。
- 追加で `W_std` / `mean` / `std` も返してよい（保存用）。

### 2.3 ルート3：関数値RFM（m×m解法）
関数値特徴を m 個:
- `φ_j(a) ∈ R^s`（格子関数）
予測:
- `ŷ(a) = Σ_{j=1..m} α_j φ_j(a)`

学習:
- `min_α Σ_i || Σ_j α_j φ_j(a_i) - y_i ||^2 + λ ||α||^2`
正規方程式:
- `(G + λ I) α = b`
- `G_{jℓ} = Σ_i <φ_j(a_i), φ_ℓ(a_i)>_Y`
- `b_j = Σ_i <φ_j(a_i), y_i>_Y`
ここで内積 `<f,g>_Y = Σ_n f_n g_n`（Δxを掛けなくてもOK。スケールはλに吸収される）

実装上はサンプルごとに
- `Φ_i ∈ R^{m×s}`（Φ_i[j,:] = φ_j(a_i)）
として
- `G += Φ_i Φ_i^T`（m×m）
- `b += Φ_i y_i`（m）

---

## 3. 実装要件（ファイル別の具体的タスク）
**重要**: 既存CLIの互換性を壊さないこと（新機能OFFなら従来と同じ挙動）。

### 3.1 `pol/features_1d.py` を拡張（Obs: fourier/proj）
#### 変更1: `build_sensor_indices` を一般化
- 既存は `obs='full'` なら 0..s-1 の index、`obs='points'` なら点センサindexを返す。
- 拡張後:
  - `obs='full'` : `torch.arange(s, long)` を返す（互換のため）
  - `obs='points'`: 従来通り `LongTensor(J,)`
  - `obs='fourier'`: `LongTensor(J,)` を返す（モード index 0..J-1）
    - `J` の上限は `max_modes = s//2 + 1`
  - `obs='proj'`: `FloatTensor(J,s)` を返す（投影行列 Ψ）
    - 乱数は `sensor_seed` を使い、`torch.Generator(device="cpu")` で固定
    - スケールは `1/sqrt(s)`（各射影が O(1) スケールになりやすい）

`proj` と `fourier` では `sensor_mode` は実質不要だが、関数シグネチャは維持し、
- `fourier`: `sensor_mode/sensor_seed` を無視
- `proj`: `sensor_seed` のみ使用、`sensor_mode` 無視
とする。

#### 変更2: `collect_observations(states, obs, sensor_idx)` を拡張
- `states` は `List[Tensor(B,s)]`
- 追加分:
  - `fourier`:
    - `z_hat = torch.fft.rfft(z, dim=-1)`（complex）
    - `sel = z_hat.index_select(dim=-1, index=modes)`（(B,J) complex）
    - 実数化して `torch.cat([sel.real, sel.imag], dim=-1)`（(B,2J)）
  - `proj`:
    - `proj = sensor_idx.to(z.device, dtype=z.dtype)`（(J,s)）
    - `obs = z @ proj.t()`（(B,J)）

`flatten_observations` は現状の concat でよい。

---

### 3.2 `pol/ridge.py` に `fit_ridge_streaming_standardized` を追加
追加関数シグネチャ（推奨）:
```python
@torch.no_grad()
def fit_ridge_streaming_standardized(
    dataloader,
    feature_fn: Callable[[torch.Tensor], torch.Tensor],
    ridge_lambda: float,
    *,
    dtype: torch.dtype = torch.float64,
    regularize_bias: bool = False,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    ...
必須要件

feature_fn が PDEシミュレーションを含む想定なので、2パス禁止（特徴を再計算しない）。

実装は以下の手順（解析変換）で 1 パスにする:

raw 特徴 phi の x_aug=[phi,1] で Gram/Cross を蓄積:

gram += x_aug.T @ x_aug

cross += x_aug.T @ y

n = gram[-1,-1]（サンプル数）

sum_phi = gram[:d,-1]、mean = sum_phi/n

sum_sq = diag(gram[:d,:d])、var = sum_sq/n - mean^2、std = sqrt(max(var,0))

標準化座標の Gram/Cross を構成:

gram_center = gram_ff - outer(sum_phi,sum_phi)/n

gram_scaled = inv_std[:,None] * gram_center * inv_std[None,:]

gram_std[:d,:d]=gram_scaled, gram_std[-1,-1]=n（他は0）

cross_center = cross_f - mean[:,None]*cross_b[None,:]

cross_scaled = inv_std[:,None]*cross_center

cross_std[:d,:]=cross_scaled, cross_std[-1,:]=cross_b

標準化座標でリッジを解いて w_std を得る

raw特徴に作用する重み w に変換（predict_linear互換）

w_features = inv_std * w_std_features

w_bias = w_std_bias - (mean*inv_std)^T w_std_features

返り値の辞書には最低限 W を含める。
追加で W_std, mean, std, gram, cross, gram_std, cross_std を入れてよい（保存・デバッグ用）。

3.3 pol/__init__.py を更新

fit_ridge_streaming_standardized を export する。

3.4 reservoir_burgers_1d.py を更新
CLI変更（互換維持）

--obs の choices を ("full","points","fourier","proj") に拡張し、helpに説明を書く。

次の新引数を追加:

--standardize-features : 0/1（default 0）

--feature-std-eps : float（default 1e-6）

import を更新:

from pol.ridge import fit_ridge_streaming, fit_ridge_streaming_standardized, predict_linear

学習部分の分岐

args.standardize_features==1 のとき

fit_ridge_streaming_standardized(...) を使う（eps=args.feature_std_eps）

それ以外は従来通り fit_ridge_streaming(...)

モデル保存の拡張（任意だが推奨）

ridge_state に mean/std/W_std が含まれる場合は保存辞書にも入れる:

feature_mean, feature_std, W_out_std

注: fit_ridge_streaming_standardized が返す W は raw特徴に作用するので、
推論側（run_eval）や predict_linear の呼び出しを変えないこと。

3.5 新規スクリプト rfm_burgers_1d.py を追加（関数値RFM）
目的

reservoir_burgers_1d.py の「ベクトル特徴→行列readout」ではなく、
関数値特徴 F(x)=(B,m,s) を作り、m×m の線形方程式で α を学習する別ルートを実装。

仕様（必須）

データ読み込み CLI と --dry-run は reservoir_burgers_1d.py と同等の流儀で実装する。

主要CLI:

データ: --data-mode, --data-file, --train-file, --test-file, --train-split, --seed, --shuffle, --ntrain, --ntest, --sub, --batch-size, --dry-run

リザーバ: --reservoir, --Tr, --dt, --ks-dt, --K, --feature-times, --rd-*, --res-burgers-nu, --ks-dealias

エンコード: --input-scale, --input-shift

RFM: --m, --rfm-activation (tanh|relu|identity), --rfm-seed, --rfm-weight-scale, --rfm-bias-scale

rfm-weight-scale<=0 のとき 1/sqrt(K_obs) を使う

ridge: --ridge-lambda, --ridge-dtype (float32|float64)

実行: --device, --out-dir, --save-model

特徴関数（関数値）:

z0 = input_scale*x + input_shift

states = reservoir.simulate(z0, dt, Tr, obs_steps)

Z = stack(states, dim=1) -> (B,K_obs,s)

固定乱数 A(m,K_obs), b(m) を用意

mixed = einsum("bks,mk->bms", Z, A) + b.view(1,m,1)

activation を点ごとに適用 -> F(x)=(B,m,s)

学習（m×m）:

gram += einsum("bms,bns->mn", F, F)（m×m）

rhs += einsum("bms,bs->m", F, y)（m）

alpha = solve((gram+λI), rhs)（Cholesky推奨）

予測:

yhat = einsum("m,bms->bs", alpha, F)

保存:

alpha, A_time_mix, b_time_mix, obs_times, obs_steps, config, reservoir_config

可視化:

既存 viz_utils.py の plot_error_histogram, plot_1d_prediction を reservoir_burgers_1d.py と同様に使って良い（失敗しても例外握りつぶし）。

4. README更新（推奨）

README.md の reservoir_burgers_1d.py セクションを更新:

--obs full|points|fourier|proj に更新

--standardize-features, --feature-std-eps を追加

実行例を追加（dry-runでOK）:

--obs fourier --J 16 --standardize-features 1

--obs proj --J 16 --sensor-seed 1 --standardize-features 1

rfm_burgers_1d.py の説明と実行例を追加

5. テスト（必須）

pytest が通ること。

既存 tests/test_reservoir_smoke.py を以下に拡張する:

obs=fourier の shape テスト

s=128 のとき max_modes=65、例えば J=8 なら各時刻 (B,16)、flatten で (B, K*16)

obs=proj の shape テスト

build_sensor_indices が (J,s) float を返すこと

fit_ridge_streaming_standardized の整合性テスト

小さなダミー特徴 phi = x[:, :10] で

pred_raw = predict_linear(phi, W_raw)

pred_std = predict_linear((phi-mean)/(std+eps), W_std)

pred_raw と pred_std が十分近いこと（allclose）

テストは外部データ不要で動くこと（dry-run相当の乱数データのみ）。

6. 受け入れ条件（Definition of Done）

python -m pytest -q が成功する。

python reservoir_burgers_1d.py --dry-run --ntrain 8 --ntest 4 --sub 256 --obs fourier --J 16 --standardize-features 1 がクラッシュせず実行できる。

python reservoir_burgers_1d.py --dry-run --ntrain 8 --ntest 4 --sub 256 --obs proj --J 16 --sensor-seed 1 --standardize-features 1 がクラッシュせず実行できる。

python rfm_burgers_1d.py --dry-run --ntrain 8 --ntest 4 --sub 256 --m 32 --K 3 --Tr 0.1 --dt 0.01 がクラッシュせず実行できる。

新機能を OFF にした場合（標準化OFF、obsがfull/points）は従来の挙動が変わらない。

新規追加/変更ファイルに __pycache__/ や .pytest_cache/ などの生成物をコミットしない。

7. 実装メモ（Codex向け）

dtype/precision:

リッジ解法は --ridge-dtype float64 をデフォルトにし、Gram/Cross/Cholesky はその dtype で計算する。

device:

観測のための index / proj 行列は .to(device) でGPUに移して良い。

乱数生成は torch.Generator(device="cpu") を使い、生成後に .to(device) する（再現性が高い）。

fourier観測は complex なので、必ず Re/Im を実数結合して返すこと。
