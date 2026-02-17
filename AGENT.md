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
