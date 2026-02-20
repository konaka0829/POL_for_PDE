# AGENT.md — Theme 1 実装: PDE-Reservoir Random Operator Features（Random Features KRR） for 1D Burgers（backprop-free）

## 0. このリポジトリの現状（必ず把握）
- 本repoはFNO（Fourier Neural Operator）元実装に加え、**backprop-free の PDE リザーバ +（任意で）ELM + Ridge** の最小実装が追加済み。
- 追加済みの主要ファイル:
  - `reservoir_burgers_1d.py`：**単一の固定リザーバPDE**を回して特徴を作り、最後を ridge 回帰（XtX/XtY 蓄積 + Cholesky）で学習するスクリプト。
  - `pol/`：リザーバPDEソルバ・特徴抽出・ELM・ridge を提供
    - `pol/reservoir_1d.py`（reaction_diffusion / ks / burgers の 1D 周期スペクトルソルバ）
    - `pol/features_1d.py`（観測時刻・センサー・flatten）
    - `pol/elm.py`（固定ランダム ELM）
    - `pol/ridge.py`（streaming ridge）
  - `tests/test_reservoir_smoke.py`：単一リザーバのスモークテスト
- 重要: **既存FNO系スクリプト（`fourier_*.py`）や既存の `reservoir_burgers_1d.py` を壊さないこと。**

---

## 1. 本タスクの目的（Theme 1）
### 1.1 Theme 1（ランダムPDE特徴 / Random Features KRR）の要点
> 「固定PDEを1つ回す」のではなく、**パラメータ θ をランダムに変えた既知PDEを R 個**用意し、  
> それらを **並列（実装はループでOK）** に回して得た観測を全部結合して特徴にする。  
> 学習は最後の線形 readout のみ（ridge 回帰）で **backprop-free** のまま。

### 1.2 追加する成果物
- 新規スクリプト（推奨）:
  - `reservoir_random_features_burgers_1d.py`  
    → Theme 1 のメイン実行スクリプト（Burgers ターゲット: a→u）
- 追加/更新するモジュール:
  - `pol/theme1_random_features_1d.py`（サンプル θ の生成、複数リザーバ特徴マップ）
  - `pol/reservoir_1d.py`（KS のパラメータ化拡張：θ を meaningful にする）
  - `pol/__init__.py`（必要なら export 追加）
- テスト:
  - `tests/test_theme1_random_features_smoke.py`（軽いスモーク）
- README:
  - Theme 1 の使い方を短く追記（例コマンド）

---

## 2. 数式仕様（Theme 1 完全定義：実装の迷いを消す）
### 2.1 土台（離散化）
- 空間: x ∈ [0,1]、周期境界
- 離散点数: s = 2**13 // sub（既存の Burgers データに合わせる）
- 入力: u ∈ R^s（mat の `a` を `::sub` して得る）
- 出力: y ∈ R^s（mat の `u` を `::sub` して得る）

### 2.2 リザーバPDE族（パラメータ θ）
既知PDE族（リザーバ）:
  ∂_t z = R_θ(z),   z(0)=E(u)
- エンコード E は最も単純に:
  z(0)=E(u)=input_scale*u + input_shift

### 2.3 ランダム化（Theme 1 の核）
- θ_1,...,θ_R を分布 p(θ) からサンプルして固定:
  θ_r ~ p(θ)   (r=1..R)
- これら θ_r は **学習で更新しない**（固定）。

### 2.4 観測（multi-time + センサー）
- 観測時刻 0 < t_1 < ... < t_K ≤ Tr
- センサー（線形汎関数） ℓ_j（j=1..J）
  - `obs=full`: ℓ_j は全格子点（J=s）
  - `obs=points`: 固定インデックス集合（JはCLI指定、equispaced/random）

### 2.5 特徴写像（Theme 1）
各リザーバ r の解:
  z_r(t;u) = G_t^{θ_r}(E(u))
特徴 Φ(u) ∈ R^M,  M = R*K*J を
  Φ_(r,k,j)(u) := ℓ_j( z_r(t_k;u) )
で定義し、(r,k,j) を 1 次元に flatten して concat する。
- concat の順番は **r-major → k → j** に統一して保存/再現可能にする。

### 2.6 学習（ridge 回帰）
線形 readout:
  ŷ(u) = W Φ(u) + b,   W∈R^{q×M}, b∈R^q（Burgers では q=s）
学習（リッジ）:
  (W*,b*) = argmin_{W,b} Σ_i ||WΦ(u_i)+b - y_i||_2^2 + λ||W||_F^2

実装は既存 `pol/ridge.fit_ridge_streaming` を利用し、
- x_aug = [features, 1]
- Gram = x_aug^T x_aug, Cross = x_aug^T y をミニバッチで蓄積
- (Gram + λI) を Cholesky で解く（逆行列は作らない）

### 2.7 Random Features KRR としての位置づけ（理論メモ：実装は primal のままでOK）
Theme 1 が誘導する有限次元カーネル:
  k_R(u,u') := Φ(u)^T Φ(u')
            = Σ_{r=1}^R Σ_{k=1}^K Σ_{j=1}^J ℓ_j(z_r(t_k;u)) ℓ_j(z_r(t_k;u'))
R→∞ で期待カーネル:
  k(u,u') = E_{θ~p(θ)} [ Σ_{k,j} ℓ_j(z_θ(t_k;u)) ℓ_j(z_θ(t_k;u')) ]
本実装は「Random Features を用いた ridge（= KRR の primal 形）」になっている。

---

## 3. 重要: メモリ爆発を避ける設計（必須）
Theme 1 は M=R*K*J が大きくなりやすい。
さらに「concat 後に global ELM」をすると、ELM 重み A のサイズが H×M になって **致命的に巨大**になり得る。

そこで **デフォルトは “per-reservoir ELM（ブロック射影）”** にする:

各リザーバ r:
  φ_r(u) ∈ R^{KJ}（または K*s）
  h_r(u) = σ(A_r φ_r(u) + c_r) ∈ R^{H_per}
最終特徴:
  h(u) = concat_r h_r(u) ∈ R^{R*H_per}

利点:
- A_r のサイズは H_per×(KJ) で済む（global の H×(R*KJ) より R 倍小さい）

実装要件:
- `--elm-mode per_reservoir|global` を用意し、デフォルトは `per_reservoir`
- `per_reservoir` のとき `--elm-h-per` を使う（総次元は R*H_per）
- `global` のとき従来通り `--elm-h`

`--use-elm 0`（生Φで ridge）も残してよいが、
- feature 次元が大きい場合は **警告 or エラー**を出して守る（Gram サイズが (M+1)^2 になるため）。

---

## 4. 実装詳細（既存実装と整合するように）
### 4.1 リザーバPDE（既存をベースに、KS を θ で拡張）
`pol/reservoir_1d.py` は rFFT を使う 1D 周期スペクトル。
ここを拡張して KS に θ を入れる。

#### 4.1.1 reaction_diffusion（そのまま）
式:
  z_t = ν z_xx + α z - β z^3
semi-implicit Euler:
  z_hat_next = (z_hat + dt*FFT(αz-βz^3)) / (1 + dt*ν*k^2)

θ = (ν, α, β)

#### 4.1.2 burgers（そのまま）
式:
  z_t = -z z_x + ν z_xx
semi-implicit Euler:
  z_hat_next = (z_hat + dt*FFT(-z z_x)) / (1 + dt*ν*k^2)

θ = (ν)

#### 4.1.3 KS（拡張：θ を追加）
現状のコードは概ね標準KS:
  z_t = - z z_x + z_xx - z_xxxx
フーリエ線形部:
  L_hat = k^2 - k^4
semi-implicit:
  z_hat_next = (z_hat + dt*FFT(-z z_x)) / (1 - dt*L_hat)

これを係数つきに拡張し、`ReservoirConfig` に以下を追加する（デフォルト1.0で既存互換）:
- ks_nl: float = 1.0   # 非線形項の係数（-ks_nl * z z_x）
- ks_c2: float = 1.0   # k^2 の係数
- ks_c4: float = 1.0   # k^4 の係数

拡張後:
  nonlinear = - ks_nl * z * z_x
  L_hat = ks_c2 * k^2 - ks_c4 * k^4
  z_hat_next = (z_hat + dt*FFT(nonlinear)) / (1 - dt*L_hat)

これにより θ = (ks_nl, ks_c2, ks_c4)

### 4.2 θ のサンプリング（Theme 1）
- `numpy.random.default_rng(theta_seed)` を使って **決定的に**サンプルする。
- 分布は最小実装として以下で十分（ただし nu のように正の値は loguniform 推奨）:
  - uniform: U(low, high)
  - loguniform: exp(U(log(low), log(high)))  ※ low>0 必須

スクリプト側/モジュール側で、reservoir タイプごとに以下をサポート:
- reaction_diffusion:
  - rd_nu_range, rd_alpha_range, rd_beta_range（+ nu は dist 指定）
- burgers:
  - res_burgers_nu_range（+ dist 指定）
- ks:
  - ks_nl_range, ks_c2_range, ks_c4_range

サンプル結果（θ_r）は学習・推論を通して固定し、`--save-model` で必ず保存する。

---

## 5. 新規スクリプト仕様: reservoir_random_features_burgers_1d.py
### 5.1 役割
- Burgers データ（mat の `a`→`u`）を読み込む
- Theme1: θ_r を R 個サンプルしてリザーバを R 個生成
- 各バッチで特徴を作る:
  - まず各 r で PDE を解き、観測を flatten して φ_r を得る
  - デフォルト: per-reservoir ELM で h_r を作り concat（h を ridge に渡す）
- ridge 回帰を streaming で学習
- train/test の mean relL2 を出す
- `viz_utils.py` で
  - テストの relL2 ヒストグラム
  - 代表サンプルの 1D プロット
  を PNG/PDF/SVG 保存
- `--dry-run` を持ち、外部データ無しで動く（shape/NaN/保存の確認）

### 5.2 CLI（最低限これを実装）
データ周り（既存と同じ）:
- `--data-mode single_split|separate_files`
- `--data-file`, `--train-file`, `--test-file`
- `--train-split`, `--seed`, `--shuffle`
- `--ntrain`, `--ntest`, `--sub`, `--batch-size`
- `--dry-run`

リザーバ・観測（既存と同じ）:
- `--reservoir reaction_diffusion|ks|burgers`
- `--Tr`, `--dt`, `--ks-dt`
- `--K`, `--feature-times`
- `--obs full|points`, `--J`, `--sensor-mode equispaced|random`, `--sensor-seed`
- `--input-scale`, `--input-shift`
- `--ks-dealias`

Theme1 固有:
- `--R`（ランダムリザーバ数）
- `--theta-seed`（θ サンプリング用 seed）

θ 分布パラメータ（reservoir ごとに）:
- reaction_diffusion:
  - `--rd-nu-range low high`
  - `--rd-alpha-range low high`
  - `--rd-beta-range low high`
  - `--rd-nu-dist loguniform|uniform`（default: loguniform）
- burgers:
  - `--res-burgers-nu-range low high`
  - `--res-burgers-nu-dist loguniform|uniform`（default: loguniform）
- ks:
  - `--ks-nl-range low high`
  - `--ks-c2-range low high`
  - `--ks-c4-range low high`

ELM:
- `--use-elm 0|1`（default: 1）
- `--elm-mode per_reservoir|global`（default: per_reservoir）
- `--elm-h-per`（per_reservoir 用 hidden 次元）
- `--elm-h`（global 用 hidden 次元）
- `--elm-activation tanh|relu`
- `--elm-seed`, `--elm-weight-scale`, `--elm-bias-scale`

Ridge:
- `--ridge-lambda`
- `--ridge-dtype float32|float64`

実行:
- `--device auto|cpu|cuda`
- `--out-dir`
- `--save-model [path]`（省略時 `out-dir/model.pt`）

### 5.3 ログ出力（必須）
標準出力に以下を出す:
- reservoir 種別 / R / K / obs / 生特徴次元 M / 最終特徴次元 H（ELM有無で）
- train relL2 / test relL2
- elapsed seconds（任意だが推奨）

### 5.4 保存する model.pt（必須）
torch.save する辞書に最低限入れる:
- `W_out`（(H+1)×q）
- `theta_list`（サンプルした ReservoirConfig の配列：dict 化して保存可）
- `sensor_idx`（points のときは必須）
- `obs_times`, `obs_steps`
- `config`（args の dict）
- ELM を使った場合:
  - per_reservoir: `elm_weight_list`, `elm_bias_list`, `elm_activation`, `elm_mode`
  - global: `elm_weight`, `elm_bias`, `elm_activation`, `elm_mode`

---

## 6. pol/theme1_random_features_1d.py（推奨実装）
### 6.1 提供するもの
- `sample_reservoir_configs(...) -> list[ReservoirConfig]`
  - reservoir タイプと R と seed と range を入力にして、ReservoirConfig を R 個返す
- `RandomReservoirFeatureMap1D`
  - constructor で
    - solvers（Reservoir1DSolver の list）
    - obs_steps / obs mode / sensor_idx
    - ELM 設定（use_elm, mode, seeds, dims）
  - `__call__(x_batch)` が (B, H) の feature を返す

### 6.2 実装注意
- torch.no_grad() を多用（PDE solve と feature は学習しない）
- device/dtype は既存スクリプトに合わせる（PDE は float32 推奨、ridge の solve は float64 推奨）
- seed:
  - θ サンプル: theta_seed
  - ELM: elm_seed + r（per_reservoir のとき）
- 既存 `FixedRandomELM` を再利用して良い

---

## 7. テスト（必須）
`tests/test_theme1_random_features_smoke.py` を追加し、以下を確認:
- 小サイズで R 個のリザーバを作って特徴が
  - shape: (B, R*H_per)（per_reservoir ELM の場合）
  - finite
- ridge 学習と推論が動き、出力 shape が (B, s) で finite

実行時間は短く（数秒）にする。

---

## 8. 受け入れ条件（Acceptance Criteria）
以下すべて満たすこと:

1) 既存のテストが壊れない:
   - `pytest -q` が通る

2) 新規スクリプトが import/compile できる:
   - `python -m py_compile reservoir_random_features_burgers_1d.py`

3) dry-run が完走し、NaN なし・出力ファイル生成:
```bash
python reservoir_random_features_burgers_1d.py \
  --dry-run --ntrain 8 --ntest 4 --sub 256 \
  --reservoir reaction_diffusion --Tr 0.2 --dt 0.01 --K 3 \
  --obs points --J 32 \
  --R 4 --theta-seed 0 \
  --use-elm 1 --elm-mode per_reservoir --elm-h-per 16 --elm-activation tanh \
  --ridge-lambda 1e-3 --ridge-dtype float64 \
  --out-dir visualizations/theme1_dryrun --save-model
