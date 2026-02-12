# AGENT.md

## 重要: codex CLI は PDF を読めない前提
- **codex CLI 側は添付 PDF を参照できません。**
- そのため、本タスクの実装仕様は **この AGENT.md に書かれた内容** と、**参照用に clone できるレポジトリ2のコード**（特に `RFM.py`）だけで完結するようにします。
- 実装者（codex）は **PDF を開かない/開けない前提で**、ここに書いた仕様と repo2 の実装を根拠に移植・統合してください。

---

## 目的
レポジトリ1（FNO 実装群があるリポジトリ）に、レポジトリ2（vvRF / RFM 実装）の **Random Feature Model (RFM)** を移植し、
**FNO と同じ Burgers 1D データ**（`burgers_data_R10.mat` の `a`→`u`）で RFM を実行できるようにする。

加えて、FNO と性能比較ができるように、RFM 側にも **誤差表示・誤差分布・代表サンプル可視化**などのプロット機能を同等に用意する。
（レポジトリ1に既にある `viz_utils.py` を再利用すること）

---

## 前提（レポジトリ構造）
### レポジトリ1（統合先）
- `fourier_1d.py` : Burgers 1D の FNO 学習スクリプト（CLI 対応済み、`viz_utils.py` で可視化も出力）
- `cli_utils.py` : `--data-mode` / `--data-file` / `--train-file` / `--test-file` などを追加する共通関数
- `viz_utils.py` : 学習曲線、誤差ヒストグラム、1D GT vs Pred 可視化など（png/pdf/svg 保存）
- `utilities3.py` : `MatReader`, `LpLoss` など

### レポジトリ2（参照元・クローン可）
- `RFM.py` : `RandomFeatureModel` 実装（Burgers 1D）
- `utilities_module.py` : 付随ユーティリティ（`dataset_with_indices` 等）
- `train.py` : 参照用（Burgers データの downsample / `K_fine` の扱い）

> 注: レポジトリ2は **参照用に clone してよい**が、統合後のレポジトリ1は **単体で動く**（レポジトリ2への runtime 依存を作らない）。

---

## RFM の実装仕様（PDF不要・ここに書いた通りに実装）
この節は **codex が PDF を見なくても実装できる**よう、レポジトリ2 `RFM.py` の内容を言語化した仕様です。
**実装は repo2 の `RFM.py` を正とし、下記仕様と一致すること**。

### 1) ランダム特徴（1D, 周期, FFT ベース）
- 入力: `a(x)`（Burgers の初期条件）を離散化した `a ∈ R^K`（K は偶数）
- 乱数パラメータ: `g_j`（j=1..m）を **フーリエ係数空間**でサンプルする
  - `g_j` は shape `(kmax,)` の複素係数（正の周波数成分のみ、`k=1..kmax`）
  - `kmax <= K//2`。`kmax` より高周波は 0 埋めして `rfft` の長さ `K//2+1` に合わせる（`zeropad`）

- GRF の KL 係数スケーリング:
  - `kwave = [0, 1, 2, ..., K//2]`
  - `sqrt_eig(k) = sig_g * (4*pi^2*k^2 + tau_g^2)^(-al_g/2)`  （k は 1..kmax の範囲で使用）
  - `g_j_scaled(k) = sqrt(2) * sqrt_eig(k) * g_j(k)` （repo2 と同じ）

- フィルタ `chi(k)`（repo2 の `act_filter` に対応）:
  - `r = abs(nu_rf * kwave * 2*pi)`
  - `chi = ReLU( min( 2*r , (0.5 + r)^(-al_rf) ) )`
  - `chi` は `kwave` と同じ長さ（`K//2+1`）で定義し、`k=0` も含む（repo2 実装どおり）

- 1つの特徴写像（バッチ対応）の計算（repo2 `rf_batch`）:
  1. `a_ft = rfft(a)`（最後次元が `K//2+1` の複素）
  2. `conv_ft(k) = a_ft(k) * g_j_scaled(k)`（ただし k=1..kmax を使用）
  3. `conv_ft` を `zeropad` して `K//2+1` に合わせる
  4. `phi_j(a) = sig_rf * ELU( K_fine * irfft( chi * conv_ft , n=K ) )`
     - `ELU` は PyTorch の `F.elu`
     - **重要**: 係数 `K_fine` を掛ける（repo2 と挙動一致のため）
     - 出力 shape は `(K,)`（バッチなら `(nbatch, K)`、複数特徴なら `(nbatch, mbatch, K)`）

### 2) モデル出力
- 学習される係数 `alpha ∈ R^m`
- 予測:
  - `u_hat(a) = (1/m) * sum_{j=1..m} alpha_j * phi_j(a)`  
  - repo2 `predict` と同じく、特徴を `bsize_grf_test` で分割して足し上げる

### 3) 学習（正則化付き最小二乗 / KRR 形式）
- 目的は「訓練サンプル i=1..n に対して u(a_i) を近似」  
- 実装は repo2 と同じく、巨大な行列 `A`（n×m×K）を作らずに、以下を直接構築する:
  - `AstarY[j] = sum_i ∫ phi_j(a_i)(x) * y_i(x) dx`
  - `AstarA[j,l] = sum_i ∫ phi_j(a_i)(x) * phi_l(a_i)(x) dx`
  - 積分は `torch.trapz(..., dx=1/K)`（repo2 と同じ）
  - `AstarA` は最終的に対称化し、`AstarA /= m`（repo2 と同じ）

- 正則化:
  - CLI では `--lambda` を受け取り、内部では **`lamreg = ntrain * lambda`** として使う（repo2 の `train.py` と同じ）
  - 線形系:
    - `lamreg > 0`: `alpha = solve(AstarA + lamreg*I, AstarY)`
    - `lamreg == 0`: `alpha = lstsq(AstarA, AstarY)`（ただし PyTorch 2.x の `torch.linalg.lstsq` を使用）

---

## 実装タスク（必須）
### 1) レポジトリ2のRFMをレポジトリ1へ移植（PyTorch 2.x 対応）
- 新規ファイルをレポジトリ1直下に追加する：
  - `rfm_burgers_1d.py`（推奨）
    - レポジトリ2の `RFM.py` から以下を **移植**（必要なら軽微に整理）
      - `RandomFeatureModel`
      - `GaussianRFcoeff`
      - （必要なら）`InnerProduct1D`（ただし本統合では「評価」は `utilities3.LpLoss` / `viz_utils.rel_l2` で行ってもよい）
    - **PyTorch 2.x 互換**にすること：
      - `torch.lstsq` は廃止 → `torch.linalg.lstsq` へ
      - `torch.linalg.lstsq(A, b).solution` の shape に注意して squeeze する
    - repo2 の `utilities_module.DataReader` 依存は作らず、**`load_train/load_test` で torch.Tensor を受け取る**設計にする（データ読込はスクリプト側で実施）
    - docstring にこの AGENT の「RFM の実装仕様」を短く要約して書く（PDFや式番号の参照は不要）

### 2) RFM 実行用スクリプトを追加（FNOと同じデータで動く）
- 新規スクリプト `rfm_1d.py` を追加する（`fourier_1d.py` と同じ場所・同じ流儀）
- 必須要件:
  - **データ読み込みは `fourier_1d.py` と同等**
    - `cli_utils.add_data_mode_args/add_split_args/validate_data_mode_args` を利用
    - `--data-mode single_split / separate_files` をサポート
    - `--sub` による subsampling（`a` と `u` を同じ間引き）
    - `--ntrain`, `--ntest`, `--seed`, `--shuffle` をサポート
    - **K_fine**（間引き前の解像度）は、元データの `a` の最後次元長として取得し RFM に渡す
  - **RFM のハイパラ引数を CLI 化**
    - `--m`（random features 数）
    - `--kmax`（RF で使う Fourier モード上限、`<= K//2`）
    - `--lambda`（KRR 正則化 λ。内部では `lamreg = ntrain * lambda` として扱う。help に明記）
    - `--nu-rf`, `--al-rf`, `--sig-rf`
    - `--tau-g`, `--al-g`, `--sig-g`（`--sig-g` が未指定なら repo2 と同じ自動設定にする）
    - バッチ関連:
      - `--bsize-train`, `--bsize-test`
      - `--bsize-grf-train`, `--bsize-grf-test`, `--bsize-grf-sample`
    - `--cpu` フラグ（あると便利。指定時は強制 cpu）
  - **FNOと比較可能なメトリクスを計算して表示・保存**
    - `utilities3.LpLoss(size_average=False)` を用いて、FNO と同様に
      - train/test の mean relative L2
    - `torch.nn.functional.mse_loss` で train/test MSE
    - `metrics.json` などに保存（再現性のため、主要ハイパラ・seed・data設定を併記）
  - **FNO相当の可視化を出力**（`viz_utils.py` を必ず使う）
    - `plot_error_histogram` : test per-sample relL2
    - `plot_1d_prediction` : 代表サンプル（例: 0,1,2）
    - 保存先は `visualizations/rfm_1d/` 配下（FNO が `visualizations/fourier_1d` を使うのと同様）

### 3) README を更新
- `README.md` に `rfm_1d.py` の実行例と主要 CLI 引数を追加
- FNO の `fourier_1d.py` と同じ `burgers_data_R10.mat` を使うことを明記

---

## 可視化・保存仕様（FNO と揃える）
- すべての figure は `viz_utils.save_figure_all_formats` 経由で **png/pdf/svg** を保存
- `visualizations/rfm_1d/` 配下に以下を出す（最低限）:
  - `test_relL2_hist.{png,pdf,svg}`
  - `sample_000.{png,pdf,svg}`, `sample_001...`
  - `metrics.json`
- 追加で出してよいもの:
  - `al_model.npy`, `grf_g.npy`（再現性用）
  - `pred_test.npy`（必要なら）

---

## 実装上の注意（落とし穴）
- `K` は偶数が必須（FFT と `rfft/irfft` 前提）。`sub` の選び方で奇数になる場合はエラーにする。
- `kmax` は `K//2` を超えないように clamp。
- レポジトリ2由来コードの `K_fine` スケーリングを維持する:
  - `rf_batch` 内の `self.K_fine * irfft(...)` のスケールを落とさない（RFM の既存挙動と一致させる）。
- GPU メモリ: `AstarA` は (m,m) で重い。`--m` を大きくする場合は注意。
  - 小さい smoke test 用のデフォルトを用意（例: m=64〜256, kmax=16〜64）

---

## 実行・スモークテスト（必須）
以下のコマンドが **エラーなく完走し、可視化ファイルが生成されること**。

### RFM（短時間）
```bash
python rfm_1d.py \
  --data-mode single_split --data-file data/burgers_data_R10.mat \
  --ntrain 64 --ntest 16 --sub 32 \
  --m 64 --kmax 16 \
  --lambda 1e-6 \
  --nu-rf 0.31946787 --al-rf 0.1 --sig-rf 2.597733 \
  --tau-g 15.045227 --al-g 2.9943917 --sig-g 1.7861565
```

### FNO（短時間）
```bash
python fourier_1d.py \
  --data-mode single_split --data-file data/burgers_data_R10.mat \
  --ntrain 64 --ntest 16 --sub 32 \
  --epochs 2 --modes 8 --width 16
```

---

## 完了条件（Acceptance Criteria）
- [ ] `rfm_1d.py` が `burgers_data_R10.mat` を読み込み、学習（線形系 solve）→評価→可視化保存まで動く
- [ ] `--data-mode single_split` と `separate_files` の両方に対応
- [ ] `visualizations/rfm_1d/` に png/pdf/svg が出力される（ヒストグラム＋少なくとも1サンプル）
- [ ] PyTorch 2.x で動作（`torch.lstsq` を使用していない）
- [ ] 主要メトリクス（train/test relL2, MSE）がコンソール表示され、`metrics.json` に保存される
