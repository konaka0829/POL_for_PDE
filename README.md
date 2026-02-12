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

### rfm_1d.py
`rfm_1d.py` は `fourier_1d.py` と同じ Burgers 1D データ（`burgers_data_R10.mat` の `a -> u`）を使って
Random Feature Model (RFM) を学習・評価し、`visualizations/rfm_1d/` に可視化と `metrics.json` を保存します。

**実行例**
```bash
python rfm_1d.py --data-mode single_split --data-file data/burgers_data_R10.mat \
  --ntrain 64 --ntest 16 --sub 32 \
  --m 64 --kmax 16 \
  --lambda 1e-6 \
  --nu-rf 0.31946787 --al-rf 0.1 --sig-rf 2.597733 \
  --tau-g 15.045227 --al-g 2.9943917 --sig-g 1.7861565
```

**主要引数**
- データ指定: `--data-mode`, `--data-file`, `--train-file`, `--test-file`, `--train-split`, `--seed`, `--shuffle`
- データサイズ: `--ntrain`, `--ntest`, `--sub`
- RFM本体: `--m`, `--kmax`, `--lambda`（内部では `lamreg = ntrain * lambda`）
- RFフィルタ: `--nu-rf`, `--al-rf`, `--sig-rf`
- GRF: `--tau-g`, `--al-g`, `--sig-g`（未指定時は自動設定）
- バッチ: `--bsize-train`, `--bsize-test`, `--bsize-grf-train`, `--bsize-grf-test`, `--bsize-grf-sample`
- 実行デバイス: `--cpu`（指定時はCPU固定）

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
