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

## RFM ハイパーパラメータ最適化（HPO）のやり方（日本語）
このリポジトリでは、RFM 用の HPO スクリプト `tune_rfm_1d.py` / `tune_rfm_2d.py` を追加しています。
**データは変更せず**（同じ `.mat` を使ったまま）、train/val/test に分割して **validation の平均 relative L2（`utilities3.LpLoss` と同等）** を最小化するハイパーパラメータを探索します。

### 1. 全体の流れ
1. **データ読み込み**: 既存の RFM スクリプトと同じ `MatReader` を使って `a,u`（1D）や `coeff,sol`（2D）を読みます。
2. **train/val/test 分割**:
   - `single_split` の場合は元データから train/test を分け、その train からさらに val を取り出します。
   - `separate_files` の場合は train ファイルから val を取り出し、test ファイルはそのまま test に使います。
3. **HPO の探索**:
   - デフォルトは **random search**（`--search random`）です。
   - `--rf-seeds 0,1,2` のように **複数 seed** を指定すると、**val の平均（必要なら標準偏差）**で比較できます。
4. **best を選択**:
   - 最小の validation error を持つ設定を best として記録します。
5. **必要なら再学習（refit）**:
   - `--refit-best` を指定すると **train+val で再学習**して test を評価します。
6. **結果保存**:
   - `--save-results` で JSON または CSV で trial 結果と best を保存します。
   - `--save-best-model` を指定すれば best を保存可能です。

### 2. 使い方例
**1D (Burgers)**
```bash
python tune_rfm_1d.py --data-mode single_split --data-file data/burgers_data_R10.mat \
  --ntrain 1000 --ntest 200 --sub 8 --batch-size 20 --m 1024 \
  --val-split 0.2 --max-trials 40 --search random --rf-seeds 0,1,2 --device cuda \
  --refit-best --save-best-model --model-out model/rfm_1d_best.pt \
  --save-results results/hpo_rfm_1d.json
```

**2D (Darcy)**
```bash
python tune_rfm_2d.py --data-mode single_split --data-file data/darcy_data.mat \
  --ntrain 1000 --ntest 100 --r 2 --grid-size 421 --batch-size 2 --m 350 \
  --val-split 0.2 --max-trials 30 --rf-seeds 0 --device cpu \
  --refit-best --save-results results/hpo_rfm_2d.json
```

### 3. 主要オプションの意味（共通）
- `--val-split`：train データから val を取り出す割合（例: 0.2）。
- `--max-trials`：ランダムサーチ試行回数。
- `--search`：`random` / `grid`。
- `--rf-seeds`：ランダム特徴の seed をカンマ区切りで指定（平均値で比較）。
- `--save-results`：JSON/CSV の出力先。
- `--refit-best`：best で train+val 再学習して test 評価。
- `--save-best-model` / `--model-out`：best モデルの保存。

### 4. 1D で探索するハイパーパラメータ（デフォルト範囲）
- `lam`：`[0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]`（choice）
- `delta`：log-uniform `[1e-4, 1.0]`
- `beta`：uniform `[0.05, 8.0]`
- `tau_theta`：log-uniform `[1.0, 30.0]`
- `alpha_theta`：uniform `[1.0, 5.0]`

### 5. 2D で探索するハイパーパラメータ（デフォルト範囲）
- `lam`：`[1e-12, 1e-10, 1e-8, 1e-6, 1e-4]`（choice）
- `tau_theta`：log-uniform `[1.0, 30.0]`
- `alpha_theta`：uniform `[1.0, 5.0]`
- `delta_sig`：log-uniform `[0.03, 0.5]`
- `s_plus`：log-uniform `[0.01, 0.5]`（正）
- `s_minus`：log-uniform `[0.01, 0.5]`（負として扱われ、`s_plus > s_minus` を満たすよう制約）
- `eta`：log-uniform `[1e-6, 1e-3]`

### 6. 計算コスト・再現性の注意
- 2D の HPO は **GPU OOM が起きやすい**ため、`--device cpu` を推奨します。
- `--batch-size` を小さめにするとメモリ負荷が下がります。
- `--seed`（データ split 用）と `--rf-seeds`（特徴生成用）を固定すると再現性が上がります。

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
