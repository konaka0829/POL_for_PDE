# AGENT.md — Koopman‑Reservoir Operator Learning (1D Burgers, backprop‑free)

## TL;DR
このリポジトリ（FNO実装）に、**逆伝播なし**（最小二乗 / ridge のみ）で学習する
**Koopman‑Reservoir Operator** を追加する。対象は **1D Burgers のみ**。
新規スクリプト `koopman_reservoir_1d.py` を追加し、既存の `fourier_1d.py` と同様に
データ読み込み（MAT）と可視化（PNG/PDF/SVG）を行う。

---

## 目的
- PDEの時間発展を **演算子（関数→関数）**として学習するが、学習は backprop を使わず、
  **ランダム固定の encoder + 線形 Koopman + 線形 decoder を ridge で一発推定**する。
- まずは Burgers の **1-step（u0→u1）**を再現できることを最重要とする。

---

## 対象データ（Burgers）
- 既存 `fourier_1d.py` と同じMATを想定
  - `a`: 初期条件 u0(x), shape = [N, S_full]
  - `u`: 目標 u1(x), shape = [N, S_full]
- `--sub` による間引き後の空間点数: `S = 2**13 // sub`（fourier_1d.py と一致）
- 追加対応（歓迎）:
  - もし `u` が [N, T, S] の時系列を含むなら、(u_t, u_{t+1}) ペアを作ってK学習できるようにする
  - ただし “1D Burgers の枠内” を超えない

---

## モデル（最小構成）
予測は以下で行う（1-step）:

1. 測定: `m = M(u)`  （u: [S] → m: [p]）
2. Reservoir encoder: `z = E(m)` （m: [p] → z: [m_res]）
3. Koopman: `z1 ≈ K z0`
4. Decoder: `u ≈ Decode(z)` （z: [m_res] → u: [S]）

推論: `u_hat1 = Decode( K @ Encode( M(u0) ) )`

---

## 重要：基底関数（複数実装する）
このタスクでは **encoder側（測定）** と **decoder側（復元）**で使う基底を複数用意する。

### 測定基底 ψ（`--measure-basis`）
基本は内積で測定:
- `m_j = <u, ψ_j> ≈ Σ_i u(x_i) ψ_j(x_i) Δx`

最低限の選択肢:
- `fourier` : 実フーリエ基底（1, cos, sin）
- `random_fourier` : ψ_j(x)=sqrt(2)cos(2π ω_j x + b_j)
- `legendre` or `chebyshev` : 多項式基底（[0,1]→[-1,1]写像）
- `rbf` : ガウスRBF（中心等間隔、幅 `--rbf-sigma`）
（任意: `sensor` = 点サンプリング）

### 復元基底 φ（`--decoder-basis`）
最低限:
- `grid` : u ≈ D z（D: [S, m_res]）
- `fourier` : u ≈ Φ c,  c = D z
- `legendre/chebyshev/rbf` のいずれか

---

## Reservoir Encoder（固定ランダム）
実装する update:
- r_{l+1} = (1-α) r_l + α tanh( W r_l + U m + b )
- z = r_{Lw}

必須CLI:
- `--reservoir-dim`, `--washout`, `--leak-alpha`
- `--spectral-radius`（Wをスケーリング）
- `--input-scale`, `--bias-scale`（任意だが推奨）

注意:
- Wのスペクトル半径調整は power iteration 等の近似でOK
- seed（`--seed`）で完全再現できるように、torch/numpy のseedを揃える

---

## Ridge 解（数値安定）
### Koopman K
目的:
- min_K ||Z1 - K Z0||_F^2 + λK ||K||_F^2

解:
- K = Z1 Z0^T (Z0 Z0^T + λK I)^{-1}

実装:
- `torch.linalg.solve` を使い、明示的 inverse を避ける
- `--ridge-k` が λK

任意（推奨）:
- `--stabilize-k` で spectral radius を推定し、必要なら K をスケールして ≤ 1 にする
  - 推定は `torch.linalg.eigvals` でも可（サイズが大きいなら power iteration）

### Decoder D
- `grid` の場合: u ≈ D z を ridge
- 基底の場合:
  1) u ≈ Φ c を ridge で解いて c を作る（係数抽出）
  2) c ≈ D z を ridge

`--ridge-d` を λD に使う。

---

## 実装ファイル方針
- 追加（必須）:
  - `koopman_reservoir_1d.py`（standalone script）
- 追加（推奨）:
  - `koopman_reservoir_utils.py`（basis生成、ridge、reservoir等を分離）
- 更新（必須）:
  - `README.md` に実行例と主要引数を追記

既存ファイル:
- `cli_utils.py` の `add_data_mode_args`, `add_split_args`, `validate_data_mode_args` を利用して
  `fourier_1d.py` と同等のCLI設計に揃える
- 可視化は `viz_utils.py` の関数を利用し、png/pdf/svg を保存する

---

## 可視化（必須）
出力先:
- `visualizations/koopman_reservoir_1d/`

保存するもの:
- test相対L2 ヒストグラム（`plot_error_histogram`）
- 代表サンプル3件の1D比較（`plot_1d_prediction`）
  - input(u0), GT(u1), Pred(u_hat1)

---

## CLI 要件（最低限）
- data:
  - `--data-mode {single_split,separate_files}`
  - `--data-file`, `--train-file`, `--test-file`
  - `--ntrain`, `--ntest`, `--sub`, `--train-split`, `--seed`, `--shuffle`
- model:
  - `--measure-basis`, `--measure-dim`
  - `--decoder-basis`, `--decoder-dim`
  - `--reservoir-dim`, `--washout`, `--leak-alpha`
  - `--spectral-radius`
  - `--ridge-k`, `--ridge-d`
  - `--basis-normalize`（推奨）
  - `--stabilize-k`（推奨）

---

## DoD（受け入れ条件）
- `python -m compileall .` が通る
- `python koopman_reservoir_1d.py --help` が通る
- 小規模設定（例: `--ntrain 20 --ntest 5 --sub 64`）で最後まで走り、
  `visualizations/koopman_reservoir_1d/` にpng/pdf/svgが生成される
- basisが少なくとも2系統切替可能で動く
  - 例: measure=fourier/decoder=grid
  - 例: measure=rbf/decoder=fourier
- README.md に実行例が追加される
- 依存追加なし（requirements.txtの範囲内）

---

## 参考：推奨の実行例（READMEにも載せる）
例:
```bash
python koopman_reservoir_1d.py \
  --data-mode single_split --data-file data/burgers_data_R10.mat \
  --ntrain 1000 --ntest 100 --sub 8 --seed 0 --shuffle \
  --measure-basis fourier --measure-dim 128 \
  --reservoir-dim 512 --washout 8 --leak-alpha 1.0 --spectral-radius 0.9 \
  --decoder-basis grid --decoder-dim 0 \
  --ridge-k 1e-6 --ridge-d 1e-6 \
  --basis-normalize --stabilize-k
```
