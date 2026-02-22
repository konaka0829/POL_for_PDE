# Koopman固有汎関数に基づく「未知PDE→既知PDEへの共役（conjugacy）学習」実装ガイド

このリポジトリ（FNO: Fourier Neural Operator）に対して、**未知PDEの時間発展（流れ）を、既知の線形PDE半群に“共役”させる座標変換**をニューラルオペレータで学習する機能を追加する。

Codex CLI（自動実装エージェント）は会話履歴を参照できないため、以下に理論仕様・データ仕様・設計・実装タスクを**漏れなく**まとめる。

---

## 0. 目的（何を追加するか）

### 追加する成果物（必須）

1. **1D版** 共役学習スクリプト: `conjugacy_1d_time.py`
2. **2D版** 共役学習スクリプト: `conjugacy_2d_time.py`
3. 共通で使える **既知PDE半群（線形時間発展）レイヤ** の実装（例: `semigroup_layers.py`）
4. 共通で使える **入出力チャネル数可変なFNOブロック**（encoder/decoderに使う。例: `fno_generic.py`）
5. README 追記（実行例・引数・データ形状）

### 実装方針（重要）

- **未知PDEはデータからのみ与えられる**（PDE式は使わない）。
- 潜在空間では **既知PDEを厳密に（あるいは高精度に）解く**（正則化ではなく “潜在空間の真の時間発展則”）。
- 学習するのは **共役座標（encoder） \(\Psi\)** と、近似逆写像（decoder） \(\Phi\approx\Psi^{-1}\)**。
- **1D と 2D の両方**で同じ思想を実装する。

---

## 1. 数式レベルの仕様（完全定義）

### 1.1 状態空間と未知PDEの流れ

- 状態空間（関数空間）: \(H\)（例: \(L^2(\Omega)\), \(H^s(\Omega)\)）
- 未知PDE（抽象形）:

\[
\partial_t u(t)=\mathcal{F}(u(t)),\quad u(0)=u_0\in H
\]

- 解作用素（流れ/半群）:

\[
S^t:H\to H,\quad S^t(u_0)=u(t),\qquad S^{t+s}=S^t\circ S^s\ (t,s\ge 0)
\]

### 1.2 既知PDE半群（潜在空間）

- 潜在空間: \(Z\)（別の関数空間。実装では同じ格子上の多チャネル場）

\[
Z \cong L^2(\Omega;\mathbb{R}^{c_z})
\]

- 既知の線形PDE（例：拡散方程式/heat）:

\[
\partial_t z(t)=A z(t),\quad A=\nu\Delta\ \ (\nu>0)
\]

- 半群:

\[
T(t)=e^{tA}:Z\to Z,\qquad z(t)=T(t)z_0
\]

**本実装ではデフォルトで `heat`（拡散）半群を採用する。**

周期境界（\(\Omega=[0,1]^d\)）ではフーリエ空間で厳密に

\[
\widehat{z}(k,t)=e^{-\nu\lVert 2\pi k\rVert^2 t}\,\widehat{z}(k,0)
\]

として計算できる（Koopman固有構造が明示的：フーリエモードが固有関数）。

### 1.3 共役（conjugacy）の定義

学習する共役写像（座標変換）:

\[
\Psi:H\to Z
\]

を用いて

\[
\boxed{\Psi(S^t u)\ \approx\ T(t)\Psi(u)\qquad (\forall t\in[0,T_{\max}])}
\]

を満たすようにする。

デコーダ \(\Phi:Z\to H\) を \(\Phi\approx\Psi^{-1}\) として同時学習し、未知PDEの予測を

\[
\hat u(t)=\Phi(T(t)\Psi(u(0)))
\]

で与える。

### 1.4 Koopman固有汎関数との関係（実装の狙いの明確化）

- Koopman作用素: \((U^t g)[u]=g(S^t u)\)
- 固有汎関数: \(U^t \varphi=e^{\lambda t}\varphi\)

線形PDE半群（例：拡散）側はフーリエモードが固有構造を持ち、共役 \(\Psi\) が存在すれば、
\(\varphi_j(u)=g_j(\Psi(u))\) は未知PDEの Koopman 固有汎関数になる。

実装上は、**潜在での時間発展が “指数減衰（各フーリエモード独立）” になるように \(\Psi\) を学ぶ**ことがこのアイデアの核。

---

## 2. 学習データ仕様（このリポジトリで扱う .mat の前提）

### 2.1 1D（Burgers系を想定）

典型データ（既存 `fourier_1d.py` と互換）:

- `a`: (N, X)  … 初期条件 \(u(t_0)\)
- `u`: (N, X)  … ある時刻 \(t_1\) の解（多くは最終時刻）

ただし、時間系列があるデータにも対応させたい:

- `u`: (N, T, X) または (N, X, T)

この場合、`a` を \(t=0\) として **先頭に prepend するオプション**を用意する。

最終的に学習スクリプト内部では

- `u_full`: (N, T_total, X)

に正規化し、ここからウィンドウ \(t_0..t_0+T\) をサンプルして学習する。

### 2.2 2D（Navier–Stokes vorticity 系を想定）

既存 `fourier_2d_time.py` と同じ形式:

- `u`: (N, S, S, T_total)（時間が最後の軸）

学習スクリプト内部では

- `u_full`: (N, S, S, T_total)

をDataLoaderで読み、各イテレーションでランダムに開始時刻 \(t_0\) を選び、
\(u(t_0..t_0+T)\) を教師系列として使う。

---

## 3. 学習目的関数（離散化・実装形）

データ \(u_i(t_k)\)（離散時刻、\(\Delta t\)一定）から、窓 \(t_0..t_0+T\) を取る。

### 3.1 予測の生成（潜在で既知PDEを解く）

- \(z_0=\Psi(u(t_0))\)
- \(z_{n+1}=T(\Delta t)\,z_n\)（n=0..T-1）
- \(\hat u(t_0+n)=\Phi(z_n)\)

### 3.2 損失（必須3項）

1) **予測損失（作用素一致）**

\[
\mathcal{L}_{pred}=\sum_{n=0}^{T}\lVert\hat u(t_0+n)-u(t_0+n)\rVert_H^2
\]

2) **半群整合（共役条件）**

\[
\mathcal{L}_{sg}=\sum_{n=0}^{T}\lVert\Psi(u(t_0+n)) - z_n\rVert_Z^2
\]

（これが \(\Psi(S^t u)\approx T(t)\Psi(u)\) の離散版）

3) **自己再構成（\(\Phi\approx\Psi^{-1}\) を支援）**

\[
\mathcal{L}_{ae}=\sum_{n=0}^{T}\lVert\Phi(\Psi(u(t_0+n))) - u(t_0+n)\rVert_H^2
\]

最終損失:

\[
\boxed{\mathcal{L}=\mathcal{L}_{pred}+\mu\,\mathcal{L}_{sg}+\lambda_{ae}\,\mathcal{L}_{ae}}
\]

### 3.3 実装上のノルム

このリポジトリ既存の `LpLoss`（相対L2）と整合させる。

- `LpLoss(size_average=False)` を使い、`myloss(pred.reshape(b,-1), gt.reshape(b,-1))` 形式で計算。
- ただし、`pred/ae` は相対L2、`sg` は潜在スケールの不定性が強いので `MSE`（絶対誤差）を推奨。
  - 推奨: `pred/ae` は `LpLoss`、`sg` は `F.mse_loss`。

---

## 4. モデル設計（1D/2D共通）

### 4.1 潜在表現

- 潜在は「同じ格子解像度」を保つ：
  - 1D: `z` shape = (B, X, c_z)
  - 2D: `z` shape = (B, S, S, c_z)

- 既知PDE半群は **各チャネル独立**（block diagonal）として適用:
  - \(A=\nu\Delta\) を各チャネルに同じように作用

### 4.2 Encoder/Decoder をFNOで実装

既存FNO実装を流用しつつ、**入出力チャネル数を可変化**する。

- 1D: `FNO1dGeneric(in_dim, out_dim, modes, width, use_grid=True)`
  - 入力は `u(x)` のチャネル `in_dim`（通常1）
  - `grid`（x座標）を concat するなら `p = nn.Linear(in_dim + 1, width)`
  - 出力 `out_dim`（encoderなら `c_z`、decoderなら1）

- 2D: `FNO2dGeneric(in_dim, out_dim, modes1, modes2, width, use_grid=True, use_instance_norm=True)`
  - grid を concat するなら `p = nn.Linear(in_dim + 2, width)`

**注意**: 既存スクリプトのFNO定義はスクリプト内に閉じている。新規追加分は

- 新しい共通モジュール（例: `fno_generic.py`）を作って import しても良い
- あるいは新規スクリプト内に定義しても良い

ただし **1D/2Dで重複が大きいので共通化推奨**。

---

## 5. 既知PDE半群レイヤ（Heat semigroup）の実装仕様

### 5.1 API

- `HeatSemigroup1d(nu, learnable_nu: bool, domain_length=1.0, use_2pi=True)`
  - `forward(z, dt)` または `apply(z, t)`
  - 入力 `z`: (B, X, C) を想定（内部で (B,C,X) にしてFFT）
  - 出力も同shape

- `HeatSemigroup2d(nu, learnable_nu: bool, domain_length=1.0, use_2pi=True)`
  - 入力 `z`: (B, S, S, C) を想定（内部で (B,C,S,S) にしてFFT2）

### 5.2 周波数グリッドと減衰因子

1D:

- `dx = L / N`（L=domain_length）
- `freq = torch.fft.rfftfreq(N, d=dx)` （cycles / unit）
- `k = 2π*freq`（use_2pi=True の場合）
- `k2 = k**2`
- 係数: `exp(-nu * k2 * dt)`

2D（rfft2 の周波数軸に注意）:

- x軸（フル）: `fx = torch.fft.fftfreq(N, d=dx)`
- y軸（rfft）: `fy = torch.fft.rfftfreq(N, d=dx)`
- `kx = 2π*fx`, `ky = 2π*fy`
- `k2 = kx[:,None]**2 + ky[None,:]**2`（shape: (N, N//2+1)）
- 係数: `exp(-nu * k2 * dt)`

### 5.3 nu の学習

- `--learn-nu` が指定されたら `raw_nu` を `nn.Parameter` として持ち、`nu = softplus(raw_nu)` で正値制約。
- 学習しない場合は float buffer として保持。

### 5.4 キャッシュ

FFT周波数グリッド（`k2`）はサイズとdeviceに依存する。

- もっとも安全: forward 内で `N` が変わる場合に再生成し、`register_buffer` ではなく dict キャッシュにする
- もしくは入力サイズ固定前提で buffer でもよい（このリポジトリは基本固定）。

---

## 6. 追加スクリプトの詳細仕様

### 6.1 `conjugacy_1d_time.py`

#### 主なCLI引数（必須）

既存スタイルに合わせること（`cli_utils.py` を利用）。

- データ:
  - `--data-mode` (`single_split` / `separate_files`)
  - `--data-file` / `--train-file` / `--test-file`
  - `--train-split`, `--seed`, `--shuffle`
- 学習:
  - `--ntrain`, `--ntest`, `--batch-size`, `--epochs`, `--learning-rate`
- 形状:
  - `--sub`（空間間引き）
  - `--T`（予測ホライズン。窓長は `T+1`）
  - `--dt`（1ステップ時間）
- モデル:
  - `--modes`, `--width`
  - `--cz`（潜在チャネル数）
  - `--mu`, `--lambda-ae`
  - `--nu`, `--learn-nu`, `--use-2pi`
- 時系列対応:
  - `--prepend-a-as-t0`（u が時系列のとき、a を先頭時刻に入れるか。デフォルト True）
  - `--random-t0`（デフォルト True。毎イテレーションで開始時刻をランダム）

#### データ整形の要点

- `a`: (N,X) → `u0`
- `u` が (N,X) の場合: `u_full = stack([a,u], dim=1)` → (N,2,X)
- `u` が (N,T,X) or (N,X,T) の場合:
  - 空間軸がどれかは `a.shape[-1]` と一致する軸で判定
  - 時間軸を真ん中に揃えて (N,T,X)
  - `prepend-a-as-t0` のとき `cat([a.unsqueeze(1), u_time], dim=1)`

#### 学習ループ（擬似コード）

```python
for batch_u_full in loader:  # (B, T_total, X)
  t0 = randint(0, T_total-(T+1))  # if random_t0 else 0
  u_gt = batch_u_full[:, t0:t0+T+1]      # (B, T+1, X)

  u0 = u_gt[:, 0].unsqueeze(-1)          # (B, X, 1)
  z = enc(u0)                             # (B, X, cz)

  loss = 0
  for n in range(T+1):
    if n > 0:
      z = heat(z, dt)
    u_hat = dec(z)                        # (B, X, 1)

    loss += LpLoss(u_hat, u_gt[:, n].unsqueeze(-1))

    z_enc = enc(u_gt[:, n].unsqueeze(-1))
    loss += mu * mse(z_enc, z)

    u_ae = dec(enc(u_gt[:, n].unsqueeze(-1)))
    loss += lambda_ae * LpLoss(u_ae, u_gt[:, n].unsqueeze(-1))

  backprop
```

### 出力（可視化）

`image/conjugacy_1d_time_<config>/` に

- 学習曲線（relL2など）
- テストサンプルの予測プロット
- エラー分布ヒストグラム

`viz_utils.py` の `plot_learning_curve`, `plot_1d_prediction`, `plot_error_histogram` を再利用する。

### 6.2 `conjugacy_2d_time.py`

#### 主なCLI引数（必須）

データ:

- `--data-mode` (`single_split` / `separate_files`)
- `--data-file` / `--train-file` / `--test-file`
- `--train-split`, `--seed`, `--shuffle`

学習:

- `--ntrain`, `--ntest`, `--batch-size`, `--epochs`, `--learning-rate`

形状:

- `--sub`（空間間引き）
- `--S`（間引き後の期待格子サイズ）
- `--T`（予測ホライズン）
- `--dt`
- 任意: `--sub-t`（時間間引き、無いなら追加して良い）

モデル:

- `--modes`, `--width`
- `--cz`, `--mu`, `--lambda-ae`, `--nu`, `--learn-nu`, `--use-2pi`
- `--use-instance-norm`（デフォルト True 推奨。既存 `fourier_2d_time.py` に倣う）

時間窓:

- `--random-t0`（デフォルト True）

#### データ整形

- `u_full`: `(N, S, S, T_total)`
- `sub` で `u_full[:, ::sub, ::sub, :]`
- `S` と一致するか `assert`。

#### 学習ループ（要点）

1Dと同様だが形が `(B,S,S,T_total)`。

- `u0 = u_gt[...,0]` を `(B,S,S,1)` にして encoder
- 予測は `pred_seq` を time last に cat して `(B,S,S,T+1)`

#### 可視化

`viz_utils.py` の

- `plot_2d_time_slices`
- `plot_rel_l2_over_time`
- `plot_error_histogram`

を使い、少数サンプルで

- `t=0, t=T/2, t=T` のスライス比較
- 時間方向 relL2 カーブ

を保存する。

## 7. 実装タスク（Codex CLI がやること）

### 7.1 新規ファイル

- `semigroup_layers.py`
- `HeatSemigroup1d`, `HeatSemigroup2d`
- `learnable nu`, `use_2pi`, `domain_length`
- 入力shape変換（`(B,X,C) ↔ (B,C,X)`, `(B,S,S,C) ↔ (B,C,S,S)`）

- `fno_generic.py`（推奨）
- `SpectralConv1d`, `SpectralConv2d`
- `MLP1d/MLP2d`
- `FNO1dGeneric(in_dim,out_dim,...)`
- `FNO2dGeneric(in_dim,out_dim,...)`
- `grid concat` オプション

- `conjugacy_1d_time.py`

- `conjugacy_2d_time.py`

### 7.2 README 追記

- 新スクリプトの説明
- 代表コマンド例
- 主要引数とデフォルト
- 入力データ `.mat` の期待形状（1D/2D）

### 7.3 既存ファイルは壊さない

`fourier_1d.py`, `fourier_2d_time.py` など既存スクリプトはそのまま動くこと。

## 8. 受け入れ条件（Done の定義）

- `python conjugacy_1d_time.py --help` と `python conjugacy_2d_time.py --help` が動く。
- データがある環境で学習が最後まで走り、損失が計算できる。
- `image/` 配下に学習曲線とサンプル可視化が保存される。
- semigroup レイヤが勾配伝播可能（nu 学習時に nu が更新される）。
- 1D/2Dともに、入力と出力のテンソルshapeが一貫している。

## 9. 推奨デフォルト値・実行例（READMEにも反映すること）

実装後、`README.md` にもこの節相当を追記する。

### 9.1 1D（`conjugacy_1d_time.py`）

推奨デフォルト（既存 `fourier_1d.py` に近い設定）:

- `--data-mode single_split`
- `--data-file data/burgers_data_R10.mat`
- `--ntrain 1000 --ntest 100`
- `--sub 8`（2^3）
- `--batch-size 20 --learning-rate 1e-3 --epochs 500`
- `--modes 16 --width 64`

共役学習特有:

- `--T 1`（u が単一スナップショットしか無い場合の最小）
- `--dt 1.0`
- `--cz 8`（または 16）
- `--mu 1.0`
- `--lambda-ae 0.1`
- `--nu 0.01`（固定でもよいし、`--learn-nu` を有効化してもよい）

実行例:

```bash
python conjugacy_1d_time.py \
  --data-mode single_split --data-file data/burgers_data_R10.mat \
  --ntrain 1000 --ntest 100 --sub 8 \
  --T 1 --dt 1.0 --cz 8 --mu 1.0 --lambda-ae 0.1 \
  --modes 16 --width 64 \
  --epochs 500 --batch-size 20 --learning-rate 1e-3
```

時間系列 `u` がある場合（例: `u` が `(N,T,X)` で格納されている）:

```bash
python conjugacy_1d_time.py \
  --data-mode single_split --data-file data/burgers_time_series.mat \
  --T 20 --dt 0.005 --random-t0 \
  --prepend-a-as-t0
```

### 9.2 2D（`conjugacy_2d_time.py`）

推奨デフォルト（既存 `fourier_2d_time.py` に近い設定）:

- `--data-mode separate_files`
- `--train-file data/ns_data_V100_N1000_T50_1.mat`
- `--test-file data/ns_data_V100_N1000_T50_2.mat`
- `--ntrain 1000 --ntest 200`
- `--sub 1 --S 64`
- `--batch-size 20 --learning-rate 1e-3 --epochs 500`
- `--modes 12 --width 20`

共役学習特有:

- `--T 40 --dt 1.0`
- `--cz 8`（または 16）
- `--mu 1.0`
- `--lambda-ae 0.1`
- `--nu 0.01`（固定でもよいし、`--learn-nu` を有効化してもよい）

実行例:

```bash
python conjugacy_2d_time.py \
  --data-mode separate_files \
  --train-file data/ns_data_V100_N1000_T50_1.mat \
  --test-file  data/ns_data_V100_N1000_T50_2.mat \
  --ntrain 1000 --ntest 200 --sub 1 --S 64 \
  --T 40 --dt 1.0 --cz 8 --mu 1.0 --lambda-ae 0.1 \
  --modes 12 --width 20 \
  --epochs 500 --batch-size 20 --learning-rate 1e-3
```
