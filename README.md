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
- 複数時刻観測で特徴 `Phi` を作成（`obs=full|points|fourier|proj`）
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
python reservoir_burgers_1d.py --dry-run --ntrain 8 --ntest 4 --sub 256 --obs fourier --J 16 --standardize-features 1
python reservoir_burgers_1d.py --dry-run --ntrain 8 --ntest 4 --sub 256 --obs proj --J 16 --sensor-seed 1 --standardize-features 1
```

**実行パターン（使い分け）**
- 通常学習: 実データを指定して実行（`--data-mode single_split` または `--data-mode separate_files`）
- 軽量確認: `--dry-run` でランダム疑似データを使い、shape/NaN/可視化出力を確認
- 保存: `--save-model` を付けると `W_out`・ELM重み・センサ情報・設定を保存

**本格実行の手順（推奨）**
1. 依存を入れる: `python3 -m pip install -r requirements.txt`
2. まず `--dry-run` で動作確認する
3. 実データ実行（単一ファイル分割）
```bash
python3 reservoir_burgers_1d.py \
  --data-mode single_split --data-file data/burgers_data_R10.mat \
  --train-split 0.8 --shuffle --seed 0 \
  --ntrain 1000 --ntest 100 --sub 8 --batch-size 20 \
  --reservoir reaction_diffusion --Tr 1.0 --dt 0.01 --K 5 \
  --obs fourier --J 64 \
  --use-elm 1 --elm-h 2048 --elm-activation tanh --elm-seed 0 \
  --ridge-lambda 1e-4 --ridge-dtype float64 \
  --standardize-features 1 --feature-std-eps 1e-6 \
  --device auto --out-dir visualizations/reservoir_burgers_full --save-model
```
4. 実データ実行（学習・評価を別ファイルで管理）
```bash
python3 reservoir_burgers_1d.py \
  --data-mode separate_files \
  --train-file data/burgers_train.mat --test-file data/burgers_test.mat \
  --ntrain 1000 --ntest 100 --sub 8 --batch-size 20 \
  --reservoir burgers --res-burgers-nu 0.05 --Tr 1.0 --dt 0.001 \
  --obs proj --J 128 --sensor-seed 1 \
  --use-elm 1 --elm-h 4096 --elm-activation relu \
  --ridge-lambda 1e-5 --ridge-dtype float64 --standardize-features 1 \
  --device auto --out-dir visualizations/reservoir_burgers_sep --save-model
```
5. `run_config.json` と可視化画像（ヒストグラム・予測プロット）で結果を確認する

**ハイパーパラメータの詳細（`reservoir_burgers_1d.py`）**
- `--ntrain`, `--ntest`:
  サンプル数を増やすほど汎化は安定しやすい。小さすぎると `W_out` が過学習しやすい。
- `--sub`:
  空間解像度 `s=2**13//sub` を決める。`sub` を小さくすると高解像度で精度向上余地はあるが、メモリと計算時間が増える。
- `--batch-size`:
  streaming ridge でも特徴計算のピークメモリを左右する。OOM時は最優先で下げる。
- `--reservoir`:
  `reaction_diffusion` は比較的安定、`ks` は高表現だが時間刻みに敏感、`burgers` はターゲット近傍の誘導がしやすい。
- `--Tr`:
  リザーバ発展時間。短すぎると特徴が浅く、長すぎると散逸で情報が薄まる。`0.3~1.0` から探索が無難。
- `--dt`, `--ks-dt`:
  時間刻み。小さいほど安定だが遅い。`ks` では `--ks-dt` を優先して `5e-4` 付近から調整するのが実用的。
- `--K`, `--feature-times`:
  観測時刻設計。`K` は等間隔、`feature-times` は任意指定。`feature-times` 指定時はその値が優先される。
- `--obs`:
  `full` は高次元で情報量最大、`points` は軽量、`fourier` は低周波重視、`proj` はランダム線形汎関数。
- `--J`:
  `points/fourier/proj` の観測次元。大きいほど表現力は上がるが計算コストも増える。`fourier` は `J <= s//2+1` 制約。
- `--sensor-mode`, `--sensor-seed`:
  `points` のセンサ配置制御。比較実験では `sensor-seed` 固定が必須。
- `--input-scale`, `--input-shift`:
  リザーバ初期条件 `z0=scale*x+shift` を規定。入力振幅を変えて非線形応答領域を調整できる。
- `--use-elm`:
  `1` で固定ランダム非線形層を使う。`0` は線形 readout 直結でベースライン比較に向く。
- `--elm-h`:
  ELM 隠れ次元。大きいほど表現力増だが `O(H^2)` のGram蓄積コストが増える。
- `--elm-activation`:
  `tanh` は滑らか、`relu` は疎で大次元時に効くことがある。
- `--elm-seed`, `--elm-weight-scale`, `--elm-bias-scale`:
  固定ランダム写像の再現性とスケール制御。`weight-scale<=0` は `1/sqrt(in_dim)` 自動設定。
- `--ridge-lambda`:
  L2正則化。小さすぎると過学習・数値不安定、大きすぎると過平滑化。`1e-6~1e-3` を対数探索するのが定石。
- `--ridge-dtype`:
  `float64` 推奨。`float32` は高速だが条件数が悪い設定で不安定化しやすい。
- `--standardize-features`, `--feature-std-eps`:
  ELM後特徴の標準化。`1` を推奨。`eps` は分母ゼロ回避で通常 `1e-6` 付近。
- `--rd-nu`, `--rd-alpha`, `--rd-beta`:
  `reaction_diffusion` の力学パラメータ。`rd-nu` は平滑化、`rd-alpha/rd-beta` は反応非線形の強さ。
- `--res-burgers-nu`:
  `burgers` リザーバ粘性。小さすぎると不安定化、大きすぎると過度に平滑化。
- `--ks-dealias`:
  KS の高周波折返し抑制。高解像度・長時間で有効。
- `--device`:
  `auto` 推奨。CPU固定比較時のみ `cpu` 明示。
- `--out-dir`, `--save-model`:
  出力管理。比較実験では run ごとに別 `out-dir` を切ると追跡しやすい。

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
- `--obs`（デフォルト: `full`）: `full|points|fourier|proj`
- `--J`（デフォルト: `128`）: `obs=points|fourier|proj` 時の次元
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
- `--standardize-features`（デフォルト: `0`）: `1` で最終特徴を学習統計で標準化して解く
- `--feature-std-eps`（デフォルト: `1e-6`）: 標準化の分母安定化項

- 実行・出力関連
- `--device`（デフォルト: `auto`）: `auto|cpu|cuda`
- `--out-dir`（デフォルト: `visualizations/reservoir_burgers_1d`）: 図・設定出力先
- `--save-model`（デフォルト: `""`）: モデル保存。引数なしで付けると `out-dir/model.pt`
- `--dry-run`（デフォルト: `False`）: 外部データ不要の検証モード

### rfm_burgers_1d.py（関数値RFM: m×m 解法）
特徴そのものを格子関数として扱い、`alpha in R^m` を解くルートです。

**dry-run（外部データ不要）**
```bash
python rfm_burgers_1d.py --dry-run --ntrain 8 --ntest 4 --sub 256 --m 32 --K 3 --Tr 0.1 --dt 0.01
```

**本格実行例**
```bash
python3 rfm_burgers_1d.py \
  --data-mode single_split --data-file data/burgers_data_R10.mat \
  --train-split 0.8 --shuffle --seed 0 \
  --ntrain 1000 --ntest 100 --sub 8 --batch-size 20 \
  --reservoir reaction_diffusion --Tr 0.5 --dt 0.005 --K 5 \
  --input-scale 1.0 --input-shift 0.0 \
  --m 256 --rfm-activation tanh --rfm-seed 0 \
  --rfm-weight-scale 0.0 --rfm-bias-scale 1.0 \
  --ridge-lambda 1e-4 --ridge-dtype float64 \
  --device auto --out-dir visualizations/rfm_burgers_full --save-model
```

**ハイパーパラメータの詳細（`rfm_burgers_1d.py`）**
- データ系 (`--data-mode`, `--data-file`, `--train-file`, `--test-file`, `--train-split`, `--seed`, `--shuffle`, `--ntrain`, `--ntest`, `--sub`, `--batch-size`):
  意味は `reservoir_burgers_1d.py` と同じ。
- リザーバ系 (`--reservoir`, `--Tr`, `--dt`, `--ks-dt`, `--K`, `--feature-times`, `--rd-*`, `--res-burgers-nu`, `--ks-dealias`):
  関数値特徴 `F(x)` を作る前段の力学を決める。時刻・刻みの設計が特徴の質に直結する。
- `--m`:
  関数値ランダム特徴の本数。学習計算量は主に `m×m` 解法に依存するため、増やしすぎると急に重くなる。
- `--rfm-activation`:
  時刻混合後の点ごとの活性化。`identity` は線形基底、`tanh/relu` は非線形基底。
- `--rfm-seed`:
  時刻混合行列 `A` とバイアス `b` の再現性を固定する。
- `--rfm-weight-scale`:
  `A` のスケール。`<=0` で `1/sqrt(K_obs)` 自動設定。大きすぎると活性化飽和を起こしやすい。
- `--rfm-bias-scale`:
  `b` のスケール。活性化の動作点を動かす。
- `--ridge-lambda`, `--ridge-dtype`:
  `alpha` 推定の安定化。`float64` + `1e-6~1e-3` 探索が安全。
- `--device`, `--out-dir`, `--save-model`, `--dry-run`:
  実行環境と出力管理。

**チューニングの実務的な順序（共通）**
1. `sub` と `batch-size` を先に決めて計算予算を固定する
2. `reservoir`, `Tr`, `dt`, `K` を粗く探索して特徴品質を決める
3. `obs/J`（または `m`）で表現次元を調整する
4. `ridge-lambda` を対数探索する
5. 最後に `ELM` 系（または `rfm-*`）のスケールを微調整する

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
