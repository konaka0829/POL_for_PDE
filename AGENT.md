# AGENT.md

## この AGENT.md の位置づけ
このファイルは、**Model123 の誤差指標と Model 1 誤差分解を改良する今回のタスク専用**です。リポジトリに既にある AGENT.md はこのタスクとは整合していないため、Codex CLI を使うときは **このファイルで repo root の AGENT.md を置き換えてから実行**してください。

---

## 1. タスクの目的

このタスクの目的は二つあります。

### 1.1 誤差指標を整理する
`pol/model123_1d/metrics.py` を、Model 1/2/3 に共通の誤差指標を定義する一元モジュールに整理してください。

主指標は **absolute discrete \(L_h^2\)** にしてください。理由は次の通りです。

- Model 1 の raw error 分解は absolute \(L^2(\mu;Y)\) 誤差の不等式として書かれている。
- Model 1/2/3 の包含関係 \(D_3\le D_2\le D_1\) も absolute 側で自然に読める。
- current setting は unit torus \([0,1)\) の一様格子なので、現在の `rms_l2` は実質的に absolute discrete \(L_h^2\) と一致する。

relative error は補助指標として残してよいですが、主指標にしないでください。

### 1.2 Model 1 の誤差分解を theorem-consistent に直す
`pol/model123_1d/error_decomposition.py` と `model1_error_decomposition_1d.py` を、TeX の Model 1 raw error decomposition の **fully discrete full-state special case** に沿って整理し直してください。

今回のスコープは **full-state special case の整理** です。一般の有限次元 Model 1、`Q_J`、`Delta_obs^(J)` の実装までは行いません。

---

## 2. 理論上の基準

### 2.1 Model 1 raw error decomposition
今回合わせたい式は
\[
D_1(\theta,\tilde T)
\le
 e^{\beta T}\Delta_{\mathrm{init}}(\theta)
 + c_{\beta,T}\Delta_{\mathrm{dyn}}(\theta;T)
 + \Delta_{\mathrm{time}}(\theta;T,\tilde T)
\]
です。

full-state かつ `E=I`, `Q=I`, `\tilde T=T` なら
\[
\Delta_{\mathrm{init}}=0,
\qquad
\Delta_{\mathrm{time}}=0,
\qquad
D_1(\theta,T)\le c_{\beta,T}\Delta_{\mathrm{dyn}}(\theta;T)
\]
に簡約されます。

### 2.2 fully discrete な評価量
fully discrete な natural quantity は次です。

- 格子幅 \(h = 1/n_x\)
- 離散内積
  \[
  \langle v_h, w_h\rangle_h = h\sum_j v_{h,j} w_{h,j}
  \]
- 離散ノルム
  \[
  \|v_h\|_{L_h^2} = \left(h\sum_j v_{h,j}^2\right)^{1/2}
  \]

また empirical quantity は
\[
D_{1,h,N}(\theta,T)^2
=
\frac1N\sum_{i=1}^N \|u_{h,i}^{N_T} - r_{h,i}^{N_T}\|_{L_h^2}^2
\]
とし、trajectory-averaged defect は
\[
\Delta_{\mathrm{dyn},h,N}(\theta;T)^2
\approx
\frac1N\sum_{i=1}^N \sum_{n=0}^{N_T} w_n \|R_{\theta,h}(r_{h,i}^n)\|_{L_h^2}^2
\]
の形にしてください。時間積分の既定値は trapezoidal rule にしてください。

### 2.3 beta の意味
\(\beta\) は **target generator の one-sided Lipschitz 定数** です。target は Burgers なので、beta の calibration では target の離散 generator
\[
F_h(z) = \nu_* z_{xx} - z z_x
\]
を使ってください。

Burgers target の safe choice は
\[
\widehat M_{\mathcal K}
=
\max\{\|(u_i^n)_x\|_{L^\infty},\ \|(r_i^n)_x\|_{L^\infty}\}
\]
から
\[
\widehat\beta_{\mathrm{safe}} = \frac12 \widehat M_{\mathcal K}
\]
です。

pairwise empirical mode では
\[
q_h(\eta^a, \eta^b)
=
\frac{\langle F_h(\eta^a)-F_h(\eta^b),\ \eta^a-\eta^b\rangle_h}{\|\eta^a-\eta^b\|_{L_h^2}^2}
\]
を多数の状態対で計算し、その最大値を使ってください。有限個の状態対しか見ないので、必要なら margin を足せるようにしてください。

---

## 3. 現状の問題点

### 3.1 metrics が分散している
- `pol/model123_1d/metrics.py` は `rms_l2` しか持たない。
- `pol/model123_1d/experiments.py` はこの `rms_l2` を使う。
- `model123_burgers_1d.py` は自前で samplewise relative error を計算している。
- `pol/model123_1d/error_decomposition.py` は別の `discrete_l2_h` を持っている。

このため、主指標と補助指標が整理されていません。

### 3.2 error decomposition の summary が theorem-consistent ではない
現状の `summary_rows["rhs_beta"]` は、実質的に
\[
\sqrt{\frac1N\sum_i (c_{\beta,T}\Delta_{\mathrm{dyn},i} + \Delta_{\mathrm{time},i})^2}
\]
に近い量です。しかし theorem-consistent に比較したいのは
\[
 e^{\beta T}\Delta_{\mathrm{init},N}
 + c_{\beta,T}\Delta_{\mathrm{dyn},N}
 + \Delta_{\mathrm{time},N}
\]
です。ここを直してください。

### 3.3 beta_mode が誤解を招く
現状の `beta_mode` は「beta の求め方」ではなく、保存する散布図の種類に近い意味になっています。これを直してください。

### 3.4 current code は full-state special case なのに、それがコード上で明示されていない
今回の error decomposition は `Q=I` の full-state special case です。一般の finite-dimensional Model 1 ではありません。このことをコードと出力 schema で明確にしてください。

---

## 4. 実装スコープ

### 4.1 今回やること
- `metrics.py` を誤差指標の一元モジュールにする。
- `experiments.py` を absolute/relative の両方を shared metrics で出すようにする。
- `model123_burgers_1d.py` を absolute/relative の両方を shared metrics で出すようにする。
- `error_decomposition.py` を theorem-consistent に整理する。
- `model1_error_decomposition_1d.py` の CLI を整理する。
- テストを追加・更新する。

### 4.2 今回やらないこと
- 一般の `Q_J` を持つ finite-dimensional Model 1 実装
- `Delta_obs^(J)` の本実装
- `reservoir_burgers_1d.py`, `rfm_burgers_1d.py`, `fourier_*.py`, `lowrank_operators/*` の大改修

必要なら import 整理だけに留めてください。

---

## 5. file-by-file 実装指示

### 5.1 `pol/model123_1d/metrics.py`
このファイルを誤差指標の shared module にしてください。

最低限、次の API を用意してください。

```python
def discrete_l2h_norm(values: torch.Tensor, *, domain_length: float = 1.0) -> torch.Tensor:
    ...

def per_sample_abs_l2h_error(pred: torch.Tensor, target: torch.Tensor, *, domain_length: float = 1.0) -> torch.Tensor:
    ...

def dataset_abs_l2h_error(pred: torch.Tensor, target: torch.Tensor, *, domain_length: float = 1.0) -> float:
    ...

def per_sample_rel_l2h_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    domain_length: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    ...

def dataset_rel_l2h_mean(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    domain_length: float = 1.0,
    eps: float = 1e-12,
) -> float:
    ...
```

実装上の注意:
- current repo は 1D periodic uniform grid なので、`domain_length=1.0` を既定でよいです。
- 最後の空間軸だけを離散空間軸として扱えば十分です。
- `rms_l2` は **後方互換の alias** として残してください。意味は `dataset_abs_l2h_error` と同じにしてください。

### 5.2 `pol/model123_1d/experiments.py`
ここでは `metrics.py` からだけ誤差を計算してください。

要求:
- absolute metric と relative metric を両方計算して `metrics` dict に入れる。
- absolute metric を main として明示する。
- 既存の `E1_train`, `E1_test`, `E2_train`, ... は壊さないでください。これらは absolute metric の legacy alias として残して構いません。
- 追加 key はたとえば
  - `E1_train_abs_l2h`
  - `E1_test_abs_l2h`
  - `E1_train_rel_l2h_mean`
  - `E1_test_rel_l2h_mean`
  のように明示的な名前にしてください。

### 5.3 `model123_burgers_1d.py`
ここでも `metrics.py` を使うようにしてください。

要求:
- train/test について absolute metric と relative metric の両方を出す。
- 標準出力では absolute metric を先に出し、その後 relative metric を出す。
- `run_config.json` には
  - `train_absL2h`
  - `test_absL2h`
  - `train_relL2`
  - `test_relL2`
  を最低限保存してください。
- `train_relL2`, `test_relL2` は既存 sweep script 互換のため残してください。
- ヒストグラムは relative でもよいですが、可能なら absolute 版も追加してください。最低でも JSON schema だけは整えてください。

### 5.4 `pol/model123_1d/error_decomposition.py`
このファイルが今回の中心です。

#### 5.4.1 scope
- ここでは **full-state special case** を扱う。
- したがって現時点では `Q = I`, `E = I` とみなしてよい。
- ただし将来の拡張を見据え、`Delta_init` は field として残してよい。現在値は 0 でよい。

#### 5.4.2 trajectory representation
現在は step 1 から `N_t` までの状態しか保存していません。これを改め、**t=0 の初期状態も含めた trajectory** を扱ってください。

推奨:
- `target_states[n]` が時刻 `t_n = n*dt` の target state
- `surrogate_states[n]` が時刻 `t_n = n*dt` の surrogate state
- shape は `(N_t + 1, N, s)`
- `n=0` は初期状態

こうしておくと trapezoidal rule を自然に実装できます。

#### 5.4.3 time quadrature
`Delta_dyn` の時間積分は trapezoidal rule を既定にしてください。

推奨 helper:

```python
def make_time_quadrature_weights(num_steps: int, dt: float, rule: str = "trapezoid") -> torch.Tensor:
    ...
```

既定値は `rule="trapezoid"` でよいです。必要なら `left` もサポートして構いませんが、既定値は trap にしてください。

#### 5.4.4 beta redesign
`beta_mode` を本当の意味で beta estimator にしてください。choices は次を推奨します。

- `zero`
- `analytic_safe`
- `analytic_safe_poincare`
- `empirical_pairwise`
- `fixed`

別 helper を作ってください。

```python
def compute_beta(
    *,
    calibration_target_states: torch.Tensor,
    calibration_surrogate_states: torch.Tensor,
    cfg: ErrorDecompositionConfig,
) -> tuple[float, dict[str, Any]]:
    ...
```

戻り値の `details` には、少なくとも mode, chosen beta, `M_K_hat` or pairwise max などの diagnostic を入れてください。

#### 5.4.5 beta の具体計算

##### zero
\[
\beta = 0
\]

##### analytic_safe
1. calibration trajectory 上の全 state について spectral derivative で \(u_x\) を計算する。
2. 各状態で格子点最大値
   \[
   \|u_x\|_{L_h^\infty} = \max_j |(u_x)_j|
   \]
   を取る。
3. target trajectory と surrogate trajectory の両方を含めた全サンプル・全時刻で最大を取り、
   \[
   \widehat M_{\mathcal K} = \max \|u_x\|_{L_h^\infty}
   \]
   とする。
4. 
   \[
   \widehat\beta = \frac12 \widehat M_{\mathcal K}
   \]
   とする。

##### analytic_safe_poincare
- `analytic_safe` と同じ `M_K_hat` を計算した上で
  \[
  \widehat\beta = -\nu_*(2\pi/L)^2 + \frac12 \widehat M_{\mathcal K}
  \]
  とする。current repo は `L=1` でよい。
- この mode は、compared states の mean が揃っている場合しか安全ではありません。したがって、**samplewise mean が target/surrogate で一致しているかを tolerance 付きで検証**し、一致しないなら `ValueError` を出してください。

##### empirical_pairwise
- calibration state pool から多数の状態対を取り、
  \[
  q_h(\eta^a,\eta^b)
  =
  \frac{\langle F_h(\eta^a)-F_h(\eta^b),\eta^a-\eta^b\rangle_h}{\|\eta^a-\eta^b\|_{L_h^2}^2}
  \]
  を計算する。
- `F_h` は **target Burgers generator** `burgers_generator(..., nu=cfg.target_nu)` を使う。
- 最大値に optional margin `cfg.beta_pairwise_margin` を足して selected beta とする。
- denominator が小さすぎる pair は skip する。

##### fixed
- `cfg.beta_fixed` をそのまま使う。

#### 5.4.6 calibration split
可能なら以下の config を追加してください。

- `calibration_num_samples: int = 0`
- `calibration_seed: int | None = None`

意味:
- `calibration_num_samples <= 0` なら evaluation sample をそのまま calibration に使う。
- `calibration_num_samples > 0` なら別 seed で別の initial condition を生成して beta calibration 専用に使う。

これは optional ですが、できるだけ入れてください。

#### 5.4.7 per-sample rows と summary rows
per-sample rows と aggregate rows を明確に分けてください。

##### per-sample row に必須の field
- `sample_index`
- `Ttilde`
- `D1_abs_l2h`
- `matched_time_error_abs_l2h`
- `Delta_init_abs_l2h`
- `Delta_dyn_abs_l2h`
- `Delta_time_abs_l2h`
- `matched_plus_time_abs_l2h`
- `rhs_beta0_pathwise_abs_l2h`
- `rhs_beta_pathwise_abs_l2h`
- `matched_rhs_beta_pathwise_abs_l2h`
- `beta_mode`
- `beta_value`
- `c_beta_T`

##### summary row に必須の field
`summary_rows` は theorem-consistent な aggregate quantity を返してください。

- `Ttilde`
- `num_samples`
- `D1`
- `matched_time_error`
- `Delta_init`
- `Delta_dyn`
- `Delta_time`
- `matched_rhs_beta`
- `rhs_beta0`
- `rhs_beta`
- `triangle_matched_plus_time`
- `beta_mode`
- `beta_value`
- `c_beta_T`

ここで意味は次の通りです。

- `D1` は
  \[
  \left(\frac1N\sum_i D_{1,i}^2\right)^{1/2}
  \]
- `Delta_dyn` は
  \[
  \left(\frac1N\sum_i \Delta_{\mathrm{dyn},i}^2\right)^{1/2}
  \]
- `Delta_time` も同様
- `rhs_beta` は
  \[
  e^{\beta T}\Delta_{\mathrm{init}} + c_{\beta,T}\Delta_{\mathrm{dyn}} + \Delta_{\mathrm{time}}
  \]
- `rhs_beta0` は
  \[
  e^{0\cdot T}\Delta_{\mathrm{init}} + \sqrt{T}\Delta_{\mathrm{dyn}} + \Delta_{\mathrm{time}}
  \]
- `matched_rhs_beta` は
  \[
  e^{\beta T}\Delta_{\mathrm{init}} + c_{\beta,T}\Delta_{\mathrm{dyn}}
  \]
- `triangle_matched_plus_time` は
  \[
  \text{matched_time_error} + \Delta_{\mathrm{time}}
  \]

必要なら legacy diagnostic として `rhs_beta_rms_of_pathwise_sum` などを追加して構いませんが、**`rhs_beta` という名前は theorem-consistent aggregate quantity にしてください。**

#### 5.4.8 aggregate_metric_rows
この関数はテストから直接呼ばれているので残してください。ただし意味を theorem-consistent な summary に直してください。

#### 5.4.9 plots
plot 名も誤解がないようにしてください。推奨は次です。

- `matched_time_scatter_beta0_baseline.*`
- `combined_scatter_beta0_baseline.*`
- `matched_time_scatter_selected_beta.*`
- `combined_scatter_selected_beta.*`
- `time_mismatch_scatter.*`
- `time_mismatch_envelope.*`
- `aggregate_bound_vs_ttilde.*`

scatter では pathwise quantity を使ってください。line plot `aggregate_bound_vs_ttilde` では `D1` と `rhs_beta`, `rhs_beta0` を比較してください。

### 5.5 `model1_error_decomposition_1d.py`
CLI を `error_decomposition.py` に合わせて整理してください。

推奨 CLI 引数:
- `--beta-mode zero|analytic_safe|analytic_safe_poincare|empirical_pairwise|fixed`
- `--beta-fixed`
- `--beta-pairwise-margin`
- `--beta-max-states`
- `--calibration-num-samples`
- `--calibration-seed`
- `--time-quadrature trapezoid|left`

既存の `--beta-mode correlation|empirical|both` は廃止して構いません。必要なら migration message を出してください。

---

## 6. backward compatibility の方針

完全互換である必要はありませんが、次はできるだけ壊さないでください。

- `rms_l2` import
- `run_experiment(...)["E1_train"]` などの legacy key
- `model123_burgers_1d.py` の `train_relL2`, `test_relL2`
- `aggregate_metric_rows(rows)` という関数名

ただし `summary_rows["rhs_beta"]` の意味は正してください。ここは **意味を直すことが優先** です。

---

## 7. テスト方針

### 7.1 既存テストの更新
少なくとも次のテスト群が通るようにしてください。

```bash
pytest \
  tests/test_model123_smoke.py \
  tests/test_model123_1d.py \
  tests/test_model1_error_decomposition_bounds.py \
  tests/test_model1_error_decomposition_formulas.py \
  tests/test_model1_error_decomposition_smoke.py \
  tests/test_model1_time_bug.py
```

### 7.2 新規または更新すべき内容
最低限、次をテストしてください。

1. `dataset_abs_l2h_error` が手計算と一致する。
2. `dataset_rel_l2h_mean` が手計算と一致する。
3. `rms_l2` が current setting では `dataset_abs_l2h_error` と一致する。
4. `run_experiment` の result dict に absolute/relative の両方が入る。
5. `model123_burgers_1d.py` の `run_config.json` に `train_absL2h`, `test_absL2h`, `train_relL2`, `test_relL2` が入る。
6. same-PDE sanity check:
   - `D1 == 0`
   - `Delta_dyn == 0`
   - `Delta_time == 0`
7. same-PDE + `Ttilde != T`:
   - `matched_time_error == 0`
   - `Delta_dyn == 0`
   - `D1 == Delta_time`
   - `rhs_beta == Delta_time`
8. summary row で
   - `rhs_beta == exp(beta*T)*Delta_init + c_beta_T*Delta_dyn + Delta_time`
   - `matched_rhs_beta == exp(beta*T)*Delta_init + c_beta_T*Delta_dyn`
   - `triangle_matched_plus_time == matched_time_error + Delta_time`
9. `fixed` beta mode が指定値を返す。
10. `analytic_safe` beta mode が有限値を返し、diagnostic `M_K_hat` を含む。

---

## 8. 実装順序の推奨
1. `metrics.py` を先に仕上げる。
2. `experiments.py` と `model123_burgers_1d.py` を shared metrics に移す。
3. `error_decomposition.py` の trajectory と quadrature を直す。
4. beta estimation を mode 化する。
5. summary row schema を theorem-consistent に直す。
6. plot 名と CLI を整理する。
7. テストを更新・追加する。
8. pytest を走らせる。

---

## 9. 最後に Codex が報告すべきこと
作業完了時には、次を簡潔に報告してください。

- どのファイルを変更したか
- new metric API
- 追加した JSON/CSV key
- `beta_mode` の新仕様
- 通したテスト
- もし互換性のために legacy alias を残したなら、その一覧
