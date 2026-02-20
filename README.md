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

### reservoir_burgers_1d.py（Backprop-free: Reservoir PDE + ELM + Ridge）
`AGENT.md` の仕様に対応した、Burgers（`a -> u`）向けの backprop-free 学習スクリプトです。  
既存の FNO スクリプトを変更せず、新規追加モジュール `pol/` を利用します。

**主な機能**
- リザーバPDE（`reaction_diffusion` / `ks` / `burgers`）を固定で時間発展
- 複数時刻観測で特徴 `Phi` を作成（`obs=full|points`）
- 固定ランダム写像 ELM（任意）を適用
- 最終 readout のみをリッジ回帰（`XtX/XtY` 蓄積 + 線形方程式求解）で学習
- `viz_utils.py` を使い、誤差ヒストグラム・代表サンプル図を PNG/PDF/SVG 保存

**実行例（reaction_diffusion）**
```bash
python reservoir_burgers_1d.py \
  --data-mode single_split --data-file data/burgers_data_R10.mat \
  --ntrain 1000 --ntest 100 --sub 8 --batch-size 20 \
  --reservoir reaction_diffusion --Tr 1.0 --dt 0.01 --K 5 --obs full \
  --use-elm 1 --elm-h 2048 --elm-activation tanh --elm-seed 0 \
  --ridge-lambda 1e-4 --ridge-dtype float64 \
  --out-dir visualizations/reservoir_burgers_rd --save-model
```

**実行例（KS）**
```bash
python reservoir_burgers_1d.py \
  --data-mode single_split --data-file data/burgers_data_R10.mat \
  --ntrain 1000 --ntest 100 --sub 8 --batch-size 20 \
  --reservoir ks --Tr 0.5 --dt 0.001 --ks-dt 0.0005 --K 5 \
  --obs points --J 256 --sensor-mode equispaced \
  --use-elm 1 --elm-h 2048 --elm-activation tanh \
  --ridge-lambda 1e-4 --ridge-dtype float64 \
  --out-dir visualizations/reservoir_burgers_ks
```

**実行例（Reservoir Burgers）**
```bash
python reservoir_burgers_1d.py \
  --data-mode single_split --data-file data/burgers_data_R10.mat \
  --ntrain 1000 --ntest 100 --sub 8 --batch-size 20 \
  --reservoir burgers --res-burgers-nu 0.05 --Tr 1.0 --dt 0.001 \
  --feature-times \"0.1,0.3,0.6,1.0\" --obs full \
  --use-elm 0 \
  --ridge-lambda 1e-5 --ridge-dtype float64 \
  --out-dir visualizations/reservoir_burgers_res
```

**dry-run（外部データ不要）**
```bash
python reservoir_burgers_1d.py --dry-run --ntrain 8 --ntest 4 --sub 256
```

**実行パターン（使い分け）**
- 通常学習: 実データを指定して実行（`--data-mode single_split` または `--data-mode separate_files`）
- 軽量確認: `--dry-run` でランダム疑似データを使い、shape/NaN/可視化出力を確認
- 保存: `--save-model` を付けると `W_out`・ELM重み・センサ情報・設定を保存

**引数とデフォルト値（`reservoir_burgers_1d.py`）**
- データ関連
- `--data-mode`（デフォルト: `single_split`）: `single_split` / `separate_files`
- `--data-file`（デフォルト: `data/burgers_data_R10.mat`）: `single_split` 時の入力ファイル
- `--train-file`（デフォルト: `None`）: `separate_files` 時の学習データ
- `--test-file`（デフォルト: `None`）: `separate_files` 時の評価データ
- `--train-split`（デフォルト: `0.8`）: `single_split` 時の学習割合
- `--seed`（デフォルト: `0`）: 乱数シード（分割・ELMなど）
- `--shuffle`（デフォルト: `False`）: `single_split` で分割前シャッフル
- `--ntrain`（デフォルト: `1000`）: 学習サンプル数
- `--ntest`（デフォルト: `100`）: テストサンプル数
- `--sub`（デフォルト: `8`）: 空間間引き率（`s = 2**13 // sub`）
- `--batch-size`（デフォルト: `20`）: DataLoader バッチサイズ

- リザーバPDE関連
- `--reservoir`（デフォルト: `reaction_diffusion`）: `reaction_diffusion|ks|burgers`
- `--Tr`（デフォルト: `1.0`）: リザーバ発展時間
- `--dt`（デフォルト: `0.01`）: 時間刻み
- `--ks-dt`（デフォルト: `0.0`）: KS専用dt（`>0` なら `dt` を上書き）
- `--K`（デフォルト: `5`）: 観測時刻数（等間隔）
- `--feature-times`（デフォルト: `""`）: 観測時刻を直接指定（例: `"0.1,0.2,0.5"`）
- `--rd-nu`（デフォルト: `1e-3`）: reaction-diffusion の拡散係数
- `--rd-alpha`（デフォルト: `1.0`）: reaction-diffusion の線形成長係数
- `--rd-beta`（デフォルト: `1.0`）: reaction-diffusion の三次非線形係数
- `--res-burgers-nu`（デフォルト: `0.05`）: reservoir burgers の粘性係数
- `--ks-dealias`（デフォルト: `False`）: KSで 2/3 dealiasing を有効化

- 観測（Obs）関連
- `--obs`（デフォルト: `full`）: `full|points`
- `--J`（デフォルト: `128`）: `obs=points` 時のセンサ数
- `--sensor-mode`（デフォルト: `equispaced`）: `equispaced|random`
- `--sensor-seed`（デフォルト: `0`）: `sensor-mode=random` の乱数シード

- 入力エンコード関連
- `--input-scale`（デフォルト: `1.0`）: 初期条件スケール
- `--input-shift`（デフォルト: `0.0`）: 初期条件シフト

- ELM関連
- `--use-elm`（デフォルト: `1`）: `0` で無効化（`h=Phi`）、`1` で有効化
- `--elm-h`（デフォルト: `2048`）: ELM 隠れ次元
- `--elm-activation`（デフォルト: `tanh`）: `tanh|relu`
- `--elm-seed`（デフォルト: `0`）: ELM 重み生成シード
- `--elm-weight-scale`（デフォルト: `0.0`）: `<=0` なら `1/sqrt(in_dim)` を自動使用
- `--elm-bias-scale`（デフォルト: `1.0`）: ELM バイアス初期化スケール

- Ridge関連
- `--ridge-lambda`（デフォルト: `1e-4`）: L2 正則化係数
- `--ridge-dtype`（デフォルト: `float64`）: `float32|float64`（数値安定性は `float64` 推奨）

- 実行・出力関連
- `--device`（デフォルト: `auto`）: `auto|cpu|cuda`
- `--out-dir`（デフォルト: `visualizations/reservoir_burgers_1d`）: 図・設定出力先
- `--save-model`（デフォルト: `""`）: モデル保存。引数なしで付けると `out-dir/model.pt`
- `--dry-run`（デフォルト: `False`）: 外部データ不要の検証モード

### reservoir_random_features_burgers_1d.py（Theme 1: Random Features KRR）
`AGENT.md` Theme 1 仕様に対応したスクリプトで、ランダムにサンプルした `R` 本のリザーバPDE特徴を結合し、最後を ridge で学習します。

**dry-run（外部データ不要）**
```bash
python reservoir_random_features_burgers_1d.py \
  --dry-run --ntrain 8 --ntest 4 --sub 256 \
  --reservoir reaction_diffusion --Tr 0.2 --dt 0.01 --K 3 \
  --obs points --J 32 \
  --R 4 --theta-seed 0 \
  --use-elm 1 --elm-mode per_reservoir --elm-h-per 16 --elm-activation tanh \
  --ridge-lambda 1e-3 --ridge-dtype float64 \
  --out-dir visualizations/theme1_dryrun --save-model
```

**実行例（single_split 実データ）**
```bash
python reservoir_random_features_burgers_1d.py \
  --data-mode single_split --data-file data/burgers_data_R10.mat \
  --ntrain 1000 --ntest 100 --sub 8 --batch-size 20 \
  --reservoir reaction_diffusion --Tr 1.0 --dt 0.01 --K 5 \
  --obs points --J 128 --sensor-mode equispaced \
  --R 8 --theta-seed 0 \
  --use-elm 1 --elm-mode per_reservoir --elm-h-per 128 --elm-activation tanh \
  --ridge-lambda 1e-4 --ridge-dtype float64 \
  --out-dir visualizations/theme1_run --save-model
```

**実行パターン（使い分け）**
- 軽量確認: `--dry-run` で shape/NaN/保存を確認
- 通常学習: `single_split` か `separate_files` で実データを指定
- 省メモリ寄り: `--use-elm 1 --elm-mode per_reservoir`（デフォルト）を推奨
- 生特徴を直接 ridge: `--use-elm 0`（特徴次元が大きいとメモリ負荷に注意）

**引数とデフォルト値（`reservoir_random_features_burgers_1d.py`）**
- データ関連
- `--data-mode`（デフォルト: `single_split`）: `single_split|separate_files`
- `--data-file`（デフォルト: `data/burgers_data_R10.mat`）
- `--train-file`（デフォルト: `None`）
- `--test-file`（デフォルト: `None`）
- `--train-split`（デフォルト: `0.8`）
- `--seed`（デフォルト: `0`）
- `--shuffle`（デフォルト: `False`）
- `--ntrain`（デフォルト: `1000`）
- `--ntest`（デフォルト: `100`）
- `--sub`（デフォルト: `8`）
- `--batch-size`（デフォルト: `20`）
- `--dry-run`（デフォルト: `False`）

- リザーバ・観測関連
- `--reservoir`（デフォルト: `reaction_diffusion`）: `reaction_diffusion|ks|burgers`
- `--Tr`（デフォルト: `1.0`）
- `--dt`（デフォルト: `0.01`）
- `--ks-dt`（デフォルト: `0.0`）: KS 時に `>0` なら `dt` を上書き
- `--K`（デフォルト: `5`）
- `--feature-times`（デフォルト: `""`）
- `--obs`（デフォルト: `full`）: `full|points`
- `--J`（デフォルト: `128`）
- `--sensor-mode`（デフォルト: `equispaced`）: `equispaced|random`
- `--sensor-seed`（デフォルト: `0`）
- `--input-scale`（デフォルト: `1.0`）
- `--input-shift`（デフォルト: `0.0`）
- `--ks-dealias`（デフォルト: `False`）

- Theme 1 固有
- `--R`（デフォルト: `8`）: ランダムリザーバ数
- `--theta-seed`（デフォルト: `0`）: θ サンプリング用 seed

- θ 分布関連
- `--rd-nu-range`（デフォルト: `1e-4 1e-2`）
- `--rd-alpha-range`（デフォルト: `0.5 1.5`）
- `--rd-beta-range`（デフォルト: `0.5 1.5`）
- `--rd-nu-dist`（デフォルト: `loguniform`）: `loguniform|uniform`
- `--res-burgers-nu-range`（デフォルト: `1e-3 2e-1`）
- `--res-burgers-nu-dist`（デフォルト: `loguniform`）: `loguniform|uniform`
- `--ks-nl-range`（デフォルト: `0.7 1.3`）
- `--ks-c2-range`（デフォルト: `0.7 1.3`）
- `--ks-c4-range`（デフォルト: `0.7 1.3`）

- ELM 関連
- `--use-elm`（デフォルト: `1`）
- `--elm-mode`（デフォルト: `per_reservoir`）: `per_reservoir|global`
- `--elm-h-per`（デフォルト: `128`）: `per_reservoir` 用
- `--elm-h`（デフォルト: `2048`）: `global` 用
- `--elm-activation`（デフォルト: `tanh`）: `tanh|relu`
- `--elm-seed`（デフォルト: `0`）
- `--elm-weight-scale`（デフォルト: `0.0`）: `<=0` なら `1/sqrt(in_dim)` を使用
- `--elm-bias-scale`（デフォルト: `1.0`）

- Ridge 関連
- `--ridge-lambda`（デフォルト: `1e-4`）
- `--ridge-dtype`（デフォルト: `float64`）: `float32|float64`

- 実行・出力関連
- `--device`（デフォルト: `auto`）: `auto|cpu|cuda`
- `--out-dir`（デフォルト: `visualizations/theme1_random_features_1d`）
- `--save-model`（デフォルト: `""`）: 引数なしで付けると `out-dir/model.pt`

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
