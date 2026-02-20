# Fourier Neural Operator

This repository contains the code for the paper:
- [(FNO) Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)

In this work, we formulate a new neural operator by parameterizing the integral kernel directly in Fourier space, allowing for an expressive and efficient architecture. 
We perform experiments on Burgers' equation, Darcy flow, and the Navier-Stokes equation (including the turbulent regime). 
Our Fourier neural operator shows state-of-the-art performance compared to existing neural network methodologies and it is up to three orders of magnitude faster compared to traditional PDE solvers.

It follows from the previous works:
- [(GKN) Neural Operator: Graph Kernel Network for Partial Differential Equations](https://arxiv.org/abs/2003.03485)
- [(MGKN) Multipole Graph Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2006.09535)


Follow-ups:
- [(PINO) Physics-Informed Neural Operator for Learning Partial Differential Equations](https://arxiv.org/pdf/2111.03794.pdf)
- [(Geo-FNO) Fourier Neural Operator with Learned Deformations for PDEs on General Geometries](https://arxiv.org/pdf/2207.05209.pdf)

Examples of applications:
- [Weather Forecast](https://arxiv.org/pdf/2202.11214.pdf)
- [Carbon capture and storage](https://arxiv.org/pdf/2210.17051.pdf)

## Requirements
- The current scripts target modern PyTorch with native complex/FFT support and are tested on
  Python 3.10+ with PyTorch 2.1+.
- Recommended versions:
  - Python: 3.10+
  - torch: 2.1+
  - torchvision: 0.16+ (only needed for `scripts/fourier_on_images.py`)
- Minimum versions:
  - Python: 3.10
  - torch: 2.0
  - torchvision: 0.15 (only needed for `scripts/fourier_on_images.py`)
- Legacy PyTorch 1.6 scripts have been removed from this repository.

## Major Updates:
- Dec 2022: Add an MLP per layer. Add InstanceNorm layers for fourier_2d_time. Add Cosine Annealing scheduler.
- Aug 2021: use GeLU instead of ReLU.
- Jan 2021: remove unnecessary BatchNorm layers.

## Files
The code is in the form of simple scripts. Each script shall be stand-alone and directly runnable.

- `fourier_1d.py` is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
The neural operator maps the solution function from time 0 to time 1.
- `fourier_2d.py` is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
The neural operator maps from the coefficient function to the solution function.
- `fourier_2d_time.py` is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf), 
which uses a recurrent structure to propagates in time. The neural operator maps the solution function from time `[t-10:t]` to time `t+1`.
- `fourier_3d.py` is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which takes the 2D spatial + 1D temporal equation directly as a 3D problem. The neural operator maps the solution function from time `[1:10]` to time `[11:T]`.
- The lowrank methods are similar. These scripts are the Lowrank neural operators for the corresponding settings.
- `data_generation` are the conventional solvers we used to generate the datasets for the Burgers equation, Darcy flow, and Navier-Stokes equation.

## 実行例とコマンドライン引数（日本語）
以下では各スクリプトの実行例と、主要なコマンドライン引数・デフォルト値をまとめます。

### fourier_1d.py
**実行例**
```bash
python fourier_1d.py --data-mode single_split --data-file data/burgers_data_R10.mat
python fourier_1d.py --data-mode separate_files --train-file data/burgers_train.mat --test-file data/burgers_test.mat
```

**引数とデフォルト値**
- `--data-mode`：`single_split` / `separate_files`（デフォルト: `single_split`）
- `--data-file`：単一ファイル読み込み時のデータパス（デフォルト: `data/burgers_data_R10.mat`）
- `--train-file` / `--test-file`：分割済みファイルのパス（デフォルト: `None`）
- `--train-split`：単一ファイル時の学習割合（デフォルト: `0.8`）
- `--seed`：分割用シード（デフォルト: `0`）
- `--shuffle`：分割前にシャッフル（デフォルト: `False`）
- `--ntrain`（デフォルト: `1000`）、`--ntest`（デフォルト: `100`）
- `--sub`（デフォルト: `8`）
- `--batch-size`（デフォルト: `20`）
- `--learning-rate`（デフォルト: `0.001`）
- `--epochs`（デフォルト: `500`）
- `--modes`（デフォルト: `16`）
- `--width`（デフォルト: `64`）

### fourier_2d.py
**実行例**
```bash
python fourier_2d.py --data-mode separate_files \
  --train-file data/piececonst_r421_N1024_smooth1.mat \
  --test-file data/piececonst_r421_N1024_smooth2.mat
python fourier_2d.py --data-mode single_split --data-file data/piececonst_r421_N1024_smooth1.mat
```

**引数とデフォルト値**
- `--data-mode`（デフォルト: `separate_files`）
- `--data-file`（デフォルト: `data/piececonst_r421_N1024_smooth1.mat`）
- `--train-file`（デフォルト: `data/piececonst_r421_N1024_smooth1.mat`）
- `--test-file`（デフォルト: `data/piececonst_r421_N1024_smooth2.mat`）
- `--ntrain`（デフォルト: `1000`）、`--ntest`（デフォルト: `100`）
- `--batch-size`（デフォルト: `20`）
- `--learning-rate`（デフォルト: `0.001`）
- `--epochs`（デフォルト: `500`）
- `--modes`（デフォルト: `12`）
- `--width`（デフォルト: `32`）
- `--r`（デフォルト: `5`）
- `--grid-size`（デフォルト: `421`）

### fourier_2d_time.py
**実行例**
```bash
python fourier_2d_time.py --data-mode separate_files \
  --train-file data/ns_data_V100_N1000_T50_1.mat \
  --test-file data/ns_data_V100_N1000_T50_2.mat
python fourier_2d_time.py --data-mode single_split --data-file data/ns_data_V100_N1000_T50_1.mat --train-split 0.8 --shuffle
```

**引数とデフォルト値**
- `--data-mode`（デフォルト: `separate_files`）
- `--data-file`（デフォルト: `data/ns_data_V100_N1000_T50_1.mat`）
- `--train-file`（デフォルト: `data/ns_data_V100_N1000_T50_1.mat`）
- `--test-file`（デフォルト: `data/ns_data_V100_N1000_T50_2.mat`）
- `--train-split`（デフォルト: `0.8`）
- `--seed`（デフォルト: `0`）
- `--shuffle`（デフォルト: `False`）
- `--ntrain`（デフォルト: `1000`）、`--ntest`（デフォルト: `200`）
- `--modes`（デフォルト: `12`）
- `--width`（デフォルト: `20`）
- `--batch-size`（デフォルト: `20`）
- `--learning-rate`（デフォルト: `0.001`）
- `--epochs`（デフォルト: `500`）
- `--sub`（デフォルト: `1`）
- `--S`（デフォルト: `64`）
- `--T-in`（デフォルト: `10`）
- `--T`（デフォルト: `40`）
- `--step`（デフォルト: `1`）

### fourier_3d.py
**実行例**
```bash
python fourier_3d.py --data-mode separate_files \
  --train-file data/ns_data_V100_N1000_T50_1.mat \
  --test-file data/ns_data_V100_N1000_T50_2.mat
python fourier_3d.py --data-mode single_split --data-file data/ns_data_V100_N1000_T50_1.mat --train-split 0.8 --shuffle
```

**引数とデフォルト値**
- `--data-mode`（デフォルト: `separate_files`）
- `--data-file`（デフォルト: `data/ns_data_V100_N1000_T50_1.mat`）
- `--train-file`（デフォルト: `data/ns_data_V100_N1000_T50_1.mat`）
- `--test-file`（デフォルト: `data/ns_data_V100_N1000_T50_2.mat`）
- `--train-split`（デフォルト: `0.8`）
- `--seed`（デフォルト: `0`）
- `--shuffle`（デフォルト: `False`）
- `--ntrain`（デフォルト: `1000`）、`--ntest`（デフォルト: `200`）
- `--modes`（デフォルト: `8`）
- `--width`（デフォルト: `20`）
- `--batch-size`（デフォルト: `10`）
- `--learning-rate`（デフォルト: `0.001`）
- `--epochs`（デフォルト: `500`）
- `--sub`（デフォルト: `1`）
- `--S`（デフォルト: `64`）
- `--T-in`（デフォルト: `10`）
- `--T`（デフォルト: `40`）

## RFM（Random Feature Model）実行方法と引数（日本語）
FNOと同様のデータ読み込み・評価・可視化に合わせたRFMの実行例と主要ハイパーパラメータです。

### rfm_1d.py（Burgers 1D）
**実行例**
```bash
python rfm_1d.py --data-mode single_split --data-file data/burgers_data_R10.mat
python rfm_1d.py --data-mode separate_files --train-file data/burgers_train.mat --test-file data/burgers_test.mat
```

**引数とデフォルト値**
- `--data-mode`：`single_split` / `separate_files`（デフォルト: `single_split`）
- `--data-file`：単一ファイル読み込み時のデータパス（デフォルト: `data/burgers_data_R10.mat`）
- `--train-file` / `--test-file`：分割済みファイルのパス（デフォルト: `None`）
- `--train-split`：単一ファイル時の学習割合（デフォルト: `0.8`）
- `--seed`：データ分割用シード（デフォルト: `0`）
- `--shuffle`：分割前にシャッフル（デフォルト: `False`）
- `--ntrain`（デフォルト: `1000`）、`--ntest`（デフォルト: `100`）
- `--sub`：間引き率（デフォルト: `8`）
- `--batch-size`（デフォルト: `20`）
- `--m`：ランダム特徴数（デフォルト: `1024`）
- `--lam`：リッジ回帰正則化係数（デフォルト: `0.0`）
- `--delta`：Burgersのフィルタ幅δ（デフォルト: `0.0025`）
- `--beta`：Burgersのフィルタ指数β（デフォルト: `4.0`）
- `--tau-theta`：GRFフィルタのτ（デフォルト: `5.0`）
- `--alpha-theta`：GRFフィルタのα（デフォルト: `2.0`）
- `--rf-seed`：ランダム特徴用シード（デフォルト: `0`）
- `--device`：`cpu` / `cuda` 指定（デフォルト: 自動）
- `--save-model`：学習済みRFMを保存（デフォルト: `False`）
- `--model-out`：保存先（デフォルト: `model/rfm_1d.pt`）

### rfm_2d.py（Darcy 2D）
**実行例**
```bash
python rfm_2d.py --data-mode separate_files \
  --train-file data/piececonst_r421_N1024_smooth1.mat \
  --test-file data/piececonst_r421_N1024_smooth2.mat
python rfm_2d.py --data-mode single_split --data-file data/darcy_data.mat
```

**引数とデフォルト値**
- `--data-mode`（デフォルト: `single_split`）
- `--data-file`（デフォルト: `data/darcy_data.mat`）
- `--train-file` / `--test-file`：分割済みファイルのパス（デフォルト: `None`）
- `--ntrain`（デフォルト: `1000`）、`--ntest`（デフォルト: `100`）
- `--batch-size`（デフォルト: `10`）
- `--r`：空間の間引き率（デフォルト: `1`）
- `--grid-size`：元の格子サイズ（デフォルト: `421`）
- `--m`：ランダム特徴数（デフォルト: `350`）
- `--lam`：リッジ回帰正則化係数（デフォルト: `1e-8`）
- `--tau-theta`：GRFフィルタのτ（デフォルト: `7.5`）
- `--alpha-theta`：GRFフィルタのα（デフォルト: `2.0`）
- `--s-plus`：σ_γ の上限（デフォルト: `1/12`）
- `--s-minus`：σ_γ の下限（デフォルト: `-1/3`）
- `--delta-sig`：σ_γ の遷移幅δ（デフォルト: `0.15`）
- `--eta`：heat smoothing の拡散係数η（デフォルト: `1e-4`）
- `--dt`：heat smoothing の時間刻み（デフォルト: `0.03`）
- `--heat-steps`：heat smoothing の反復回数（デフォルト: `34`）
- `--f-const`：右辺定数f（デフォルト: `1.0`）
- `--rf-seed`：ランダム特徴用シード（デフォルト: `0`）
- `--seed`：データ分割や乱数のシード（デフォルト: `0`）
- `--device`：`cpu` / `cuda` 指定（デフォルト: 自動）
- `--save-model`：学習済みRFMを保存（デフォルト: `False`）
- `--model-out`：保存先（デフォルト: `model/rfm_2d.pt`）

### scripts/eval.py
**実行例**
```bash
python scripts/eval.py --data-file data/ns_data_V1e-4_N20_T50_R256test.mat \
  --model-file model/ns_fourier_V1e-4_T20_N9800_ep200_m12_w32
```

**引数とデフォルト値**
- `--data-file`（デフォルト: `data/ns_data_V1e-4_N20_T50_R256test.mat`）
- `--model-file`（デフォルト: `model/ns_fourier_V1e-4_T20_N9800_ep200_m12_w32`）
- `--ntest`（デフォルト: `20`）
- `--sub`（デフォルト: `4`）
- `--sub-t`（デフォルト: `4`）
- `--S`（デフォルト: `64`）
- `--T-in`（デフォルト: `10`）
- `--T`（デフォルト: `20`）
- `--indent`（デフォルト: `3`）

### scripts/super_resolution.py
**実行例**
```bash
python scripts/super_resolution.py --data-file data/ns_data_V1e-4_N20_T50_test.mat \
  --model-file model/ns_fourier_V1e-4_T20_N9800_ep200_m12_w32
```

**引数とデフォルト値**
- `--data-file`（デフォルト: `data/ns_data_V1e-4_N20_T50_test.mat`）
- `--model-file`（デフォルト: `model/ns_fourier_V1e-4_T20_N9800_ep200_m12_w32`）
- `--ntest`（デフォルト: `20`）
- `--sub`（デフォルト: `1`）
- `--sub-t`（デフォルト: `1`）
- `--S`（デフォルト: `64`）
- `--T-in`（デフォルト: `10`）
- `--T`（デフォルト: `20`）
- `--indent`（デフォルト: `1`）

## Subordination 1D（Heat semigroup + Bernstein ψ）
1D 周期領域に対して、半群
`u_hat(k,t) = exp(-t * psi(lambda_k)) * a_hat(k)`
を直接学習する最小実装です。`psi` は Bernstein 関数パラメータ化で推定します。

### データ生成（分数拡散）
```bash
python data_generation/fractional_diffusion_1d/gen_fractional_diffusion_1d.py \
  --out-file data/fractional_diffusion_1d_alpha0.5.mat \
  --N 1500 --S 1024 --T 11 --t-max 1.0 --alpha 0.5 --seed 0
```

### 学習
```bash
python subordination_1d_time.py \
  --data-mode single_split --data-file data/fractional_diffusion_1d_alpha0.5.mat \
  --ntrain 1000 --ntest 200 --sub 2 --sub-t 1 \
  --batch-size 20 --learning-rate 1e-2 --epochs 300 \
  --psi-J 32 --learn-s --psi-s-min 1e-3 --psi-s-max 1e3 \
  --plot-psi
```

### ψ の解析的ベースライン推定（log-ratio）
```bash
python scripts/estimate_psi_logratio_1d.py \
  --data-mode single_split --data-file data/fractional_diffusion_1d_alpha0.5.mat \
  --sub 2 --sub-t 1 --split all \
  --viz-dir visualizations/psi_baseline_1d \
  --plot-psi-true
```

### Monte Carlo subordination 評価
```bash
python subordination_1d_time.py \
  --data-mode single_split --data-file data/fractional_diffusion_1d_alpha0.5.mat \
  --ntrain 1000 --ntest 200 --sub 2 --sub-t 1 \
  --batch-size 20 --learning-rate 1e-2 --epochs 300 \
  --psi-J 32 --learn-s --psi-s-min 1e-3 --psi-s-max 1e3 \
  --plot-psi \
  --mc-samples 256 --mc-seed 0 --mc-batch-size 20 --mc-chunk 0
```

### データフォーマット（.mat）
- `a`: `(N, S)` float32（初期条件）
- `u`: `(N, S, T)` float32（解、time-last）
- `t`: `(T,)` float32（時刻）
- 任意メタ: `alpha`, `S`, `T`, `t_max`, `grf_beta`, `grf_scale`

### 本格実行の流れ（推奨）
1. `gen_fractional_diffusion_1d.py` でデータ生成
2. `subordination_1d_time.py` で学習
3. `visualizations/subordination_1d_time/` の出力（`learning_curve`, `error_hist`, `sample_*`, `psi_curve`）を確認

実行環境によっては `python` ではなく `python3` を使ってください。

### 実装の動き方（subordination_1d_time.py）
1. `a(N,S), u(N,S,T), t(T)` を読み込み、`--sub`, `--sub-t` で間引く。
2. `--data-mode single_split` の場合は `train_split` で分割し、`ntrain/ntest` を切り出す。
3. `psi(lambda)=a0 + b*lambda + Σ alpha_j (1-exp(-s_j*lambda))` を学習する。
4. forward は FFT 対角化で `u_hat(k,t)=exp(-t*psi(lambda_k))*a_hat(k)` を計算する。
5. 損失はバッチ平均 relative L2（`||pred-gt||/||gt||`）で学習する。
6. 学習後に deterministic 予測で `learning_curve`, `error_hist`, `sample_*` を保存する。
7. `--plot-psi` かつデータに `alpha` がある場合、`psi_true=lambda^alpha` と比較して `psi_curve` を保存する。
8. `--mc-samples > 0` の場合、Poisson サンプリングによる Monte Carlo subordination を追加評価し、`mc_vs_det_hist`, `mc_vs_gt_hist` と重ね描きサンプル図を保存する。

### 学習スクリプト引数（subordination_1d_time.py、デフォルト値）
- `--data-mode`（`single_split`）
- `--data-file`（`data/fractional_diffusion_1d_alpha0.5.mat`）
- `--train-file`（`None`）
- `--test-file`（`None`）
- `--train-split`（`0.8`）
- `--seed`（`0`）
- `--shuffle`（デフォルト無効。指定時のみ有効）
- `--ntrain`（`1000`）
- `--ntest`（`200`）
- `--sub`（`1`）
- `--sub-t`（`1`）
- `--batch-size`（`20`）
- `--learning-rate`（`1e-2`）
- `--epochs`（`300`）
- `--psi-J`（`32`）
- `--learn-s`（デフォルト無効。指定時のみ `s_j` を学習）
- `--psi-s-min`（`1e-3`）
- `--psi-s-max`（`1e3`）
- `--psi-eps`（`1e-8`）
- `--viz-dir`（`visualizations/subordination_1d_time`）
- `--plot-psi`（デフォルト無効）
- `--plot-samples`（`3`）
- `--mc-samples`（`0`。`0` で MC 評価をスキップ）
- `--mc-seed`（`0`）
- `--mc-batch-size`（`0`。`0` のとき `--batch-size` を使用）
- `--mc-chunk`（`0`。`0` のとき chunk 分割なし）

### ψ ベースライン推定の動き方（scripts/estimate_psi_logratio_1d.py）
1. `a,u,t` を読み込み、`--split` で `all/train/test` を選ぶ。
2. `a_hat=rfft(a), u_hat=rfft(u)` を計算。
3. `y=log(|u_hat|+eps)-log(|a_hat|+eps)` を作る。
4. 振幅閾値 `amp-threshold` でマスクし、時刻方向の原点回帰で `psi_hat(lambda_k)` を推定。
5. `psi_hat` を 0 以上に clamp し、`k=0` は `psi_hat[0]=0` を明示。
6. `psi_baseline_curve.(png/pdf/svg)` と `psi_baseline.npz` を保存する。

### ψ ベースライン推定引数（estimate_psi_logratio_1d.py、デフォルト値）
- `--data-mode`（`single_split`）
- `--data-file`（`data/fractional_diffusion_1d_alpha0.5.mat`）
- `--train-file`（`None`）
- `--test-file`（`None`）
- `--train-split`（`0.8`）
- `--seed`（`0`）
- `--shuffle`（デフォルト無効）
- `--sub`（`1`）
- `--sub-t`（`1`）
- `--split`（`all`）
- `--amp-threshold`（`1e-8`）
- `--log-eps`（`1e-12`）
- `--max-samples`（`0`。`0` 以下で無制限）
- `--viz-dir`（`visualizations/psi_baseline_1d`）
- `--out-npz`（空文字。未指定時は `<viz-dir>/psi_baseline.npz`）
- `--plot-psi-true`（デフォルト無効）

### データ生成スクリプトの主要ハイパーパラメータ（gen_fractional_diffusion_1d.py）
- `--out-file`：出力 `.mat` ファイル
- `--N`：サンプル数
- `--S`：空間グリッド数
- `--T`：時刻数
- `--t-max`：最終時刻
- `--alpha`：分数拡散指数（真の `psi(lam)=lam^alpha`）
- `--seed`：乱数シード
- `--grf-beta`：初期条件 GRF のスペクトル減衰（大きいほど滑らか）
- `--grf-scale`：初期条件振幅スケール

### 実運用の調整目安
- まず `--psi-J 16` か `--psi-J 32` で開始
- 不安定なら `--learning-rate` を下げる（例: `1e-2 -> 5e-3 -> 1e-3`）
- 高速化したい場合は `--sub` を大きくする（ただし分解能は低下）
- `--plot-psi` を有効にして、`psi` の形状が真値と整合するか確認

## Datasets
We provide the Burgers equation, Darcy flow, and Navier-Stokes equation datasets we used in the paper. 
The data generation configuration can be found in the paper.
- [PDE datasets](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing)

The datasets are given in the form of matlab file. They can be loaded with the scripts provided in utilities.py. 
Each data file is loaded as a tensor. The first index is the samples; the rest of indices are the discretization.
For example, 
- `Burgers_R10.mat` contains the dataset for the Burgers equation. It is of the shape `[1000, 8192]`, 
meaning it has `1000` training samples on a grid of `8192`.
- `NavierStokes_V1e-3_N5000_T50.mat` contains the dataset for the 2D Navier-Stokes equation. It is of the shape `[5000, 64, 64, 50]`, 
meaning it has `5000` training samples on a grid of `(64, 64)` with `50` time steps.

We also provide the data generation scripts at `data_generation`.

## Models
Here are the pre-trained models. It can be evaluated using _eval.py_ or _super_resolution.py_.
- [models](https://drive.google.com/drive/folders/1swLA6yKR1f3PKdYSKhLqK4zfNjS9pt_U?usp=sharing)

## Citations

```
@misc{li2020fourier,
      title={Fourier Neural Operator for Parametric Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2010.08895},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{li2020neural,
      title={Neural Operator: Graph Kernel Network for Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2003.03485},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
