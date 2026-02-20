# AGENT.md — 1D Subordination-based Operator Learning (Heat semigroup + Bernstein ψ)

## 0. 目的（このリポジトリで実装する新機能）
このリポジトリ（FNOのスクリプト群）に対し、**1D のみ**を対象として、以下の「理論に根ざした」新規研究テーマの最小実装を追加する。

### 研究テーマ（1D版 / 最小実装）
- 既知PDEとして **熱方程式（heat semigroup）** を計算資源とみなし、
- 未知PDEの半群を **Bochner subordination** により
  \[
    S(t) = e^{-t\psi(-\Delta)}
  \]
  の形で表すクラスに限定し、
- データ（初期条件→複数時刻の解）から **1次元関数 ψ（Bernstein関数）** を推定する。

**重要**：これは「PDEを損失に足す（PINN的）」のではなく、  
**既知PDE（熱）の半群を核として、未知半群を functional calculus で構成する**という必然性（定理）に立脚する。

---

## 1. 数理仕様（実装に直結する定義）

### 1.1 1D 周期領域の設定
- 領域：\(\Omega = [0,1)\)（周期境界条件）
- 空間離散：等間隔グリッド \(S\) 点
- ラプラシアン：\(-\Delta\) の固有値（フーリエモード）
  \[
    \lambda_k = (2\pi k)^2,\quad k=0,1,\dots,\lfloor S/2\rfloor
  \]
  （実装では `torch.fft.rfftfreq` で \(k\) を作り、`2π` を掛けて二乗する。）

### 1.2 未知半群（subordinationクラス）
未知PDEの時間発展（半群）を
\[
  \widehat{u}(k,t) = e^{-t\psi(\lambda_k)} \widehat{a}(k)
\]
で表す。ここで
- 入力：初期条件 \(a(x)\)
- 出力：複数時刻 \(t_0,\dots,t_{T-1}\) における解 \(u(x,t_j)\)
- \(\psi:[0,\infty)\to[0,\infty)\) は **Bernstein関数**（非負・増加・完全単調な導関数などの性質）

### 1.3 Bernstein関数 ψ のパラメータ化（必須仕様）
Lévy–Khintchine 形の離散近似として、次で表現する：

\[
\psi(\lambda)=a + b\lambda + \sum_{j=1}^J \alpha_j (1-e^{-s_j\lambda})
\]
制約：
- \(a\ge 0\), \(b\ge 0\), \(\alpha_j\ge 0\), \(s_j>0\)

実装：
- `softplus` により非負制約を保証
- \(s_j\) は `logspace(s_min, s_max, J)` で初期化し、**学習する/固定する**をフラグで切替可能にする

---

## 2. 追加・変更するファイル（このリポジトリの文化に合わせる）
このリポジトリは「各スクリプトが stand-alone」なので、追加実装もその流儀に合わせる。

### 2.1 追加：学習スクリプト（必須）
- **`subordination_1d_time.py`**（新規）
  - 1Dの初期条件 `a` → 複数時刻の解 `u(t)` を学習
  - 学習対象は **ψ のみ**（小パラメータ）
  - FFT で forward を実装（複素数は torch の `rfft/irfft` を使用）
  - 既存の `cli_utils.py` を利用し、`--data-mode`（single_split / separate_files）をサポート
  - 可視化は `viz_utils.py` を利用して png/pdf/svg を保存

### 2.2 追加：データ生成（検証用・必須）
- **`data_generation/fractional_diffusion_1d/gen_fractional_diffusion_1d.py`**（新規）
  - 既知の真の \(\psi\) を持つデータ（例：分数拡散）を **FFTで厳密生成**
  - 分数拡散：
    \[
      \partial_t u = -(-\Delta)^\alpha u \quad\Rightarrow\quad \psi(\lambda)=\lambda^\alpha
    \]
  - 初期条件 \(a\) は Gaussian random field（GRF）としてスペクトル合成で生成（詳細は後述）

### 2.3 変更：可視化ユーティリティ（推奨）
- `viz_utils.py` に **ψの曲線プロット関数**を追加する：
  - `plot_psi_curve(lam, psi_pred, psi_true=None, logx=True, logy=True, title=None, out_path_no_ext=...)`
  - 既存の `save_figure_all_formats` で png/pdf/svg 保存

### 2.4 変更：README（必須）
- `README.md` に以下を追加：
  - `subordination_1d_time.py` の実行例と主要引数
  - `gen_fractional_diffusion_1d.py` の実行例
  - データフォーマット（`a`, `u`, `t`）の説明

---

## 3. データフォーマット仕様（.mat）
既存の `MatReader`（utilities3.py）で読める .mat を前提。

### 3.1 学習データ（必須フィールド）
- `a`: shape `(N, S)` float32  
  初期条件
- `u`: shape `(N, S, T)` float32  
  解（time-last を採用。2D_time と整合。）
- `t`: shape `(T,)` float32  
  時刻配列（`t[0]=0` を含めて良い）

### 3.2 メタ（任意）
- `alpha`: scalar float（分数拡散の指数）
- `S`, `T`, `t_max`, `beta`, `grf_scale` など

学習スクリプトは `t` があればそれを使う。無い場合は CLI 引数で times を与えられる設計にしてもよいが、最小実装では **`t` 必須**でOK。

---

## 4. `subordination_1d_time.py` の詳細仕様（実装要件）

### 4.1 CLI 引数（最低限）
既存の `fourier_1d.py` に揃える。

#### データ関連（既存ユーティリティ利用）
- `--data-mode {single_split,separate_files}`
- `--data-file`（single_split）
- `--train-file`, `--test-file`（separate_files）
- `--train-split`, `--seed`, `--shuffle`

#### 学習・前処理
- `--ntrain`, `--ntest`
- `--sub`（空間サブサンプリング：`a[:, ::sub]`, `u[:, ::sub, :]`）
- `--sub-t`（任意：時間サブサンプリング：`u[..., ::sub_t]`, `t[::sub_t]`）
- `--batch-size`, `--learning-rate`, `--epochs`

#### ψ モデル
- `--psi-J`（原子数 J）
- `--learn-s`（指定時 `s_j` も学習。無指定なら固定）
- `--psi-s-min`, `--psi-s-max`（logspace 範囲、例：1e-3〜1e3）
- `--psi-eps`（`s_j` の下限用 eps）

#### 可視化
- `--viz-dir`（デフォルト：`visualizations/subordination_1d_time`）
- `--plot-psi`（可能なら true ψ も同時に描く）
- `--plot-samples`（サンプル数 / 何個描画するか）

### 4.2 モデル実装（必須クラス）
- `BernsteinPsi(nn.Module)`
  - パラメータ：`log_a`, `log_b`, `log_alpha[J]`, `log_s[J]`（learn_sの場合）
  - forward: `psi(lam)` を返す（lam >= 0 のテンソル）

- `SubordinatedHeatOperator1D(nn.Module)`
  - buffer: `lam`（shape `(S//2+1,)`）
  - forward(a, t):  
    入力 `a` shape `(B,S,1)` or `(B,S)`  
    `t` shape `(T,)`  
    出力 `u_pred` shape `(B,S,T)`

**Forward の式（必須）**：
1. `a_hat = rfft(a)` → shape `(B, K)` complex, K = S//2+1
2. `psi_lam = psi(lam)` → shape `(K,)` real
3. `decay = exp(-t[:,None] * psi_lam[None,:])` → shape `(T,K)` real
4. `u_hat = a_hat[:,None,:] * decay[None,:,:]` → shape `(B,T,K)` complex
5. `u = irfft(u_hat, n=S)` → shape `(B,T,S)` real
6. `u = u.permute(0,2,1)` → shape `(B,S,T)`

### 4.3 損失と評価（最低限）
- training loss：MSE でもよいが、既存と整合するなら「相対L2」を推奨
- ただし `LpLoss` は空間次元前提なので、最小実装では以下で OK：

相対L2（バッチ平均）：
- flatten: `(B, S*T)`
- `loss = mean( ||pred-gt||2 / (||gt||2 + eps) )`

評価：
- test relative L2（全バッチ平均）

### 4.4 可視化（必須）
`viz_utils.py` を使い、png/pdf/svg を保存。

保存先：
- `visualizations/subordination_1d_time/`

必須出力：
- 学習曲線（train/test relL2）
- test の per-sample relL2 ヒストグラム
- 代表サンプルで、いくつかの `t_index` を選び `plot_1d_prediction` を複数回呼ぶ  
  （例：t=0, t=T//2, t=T-1）

推奨：
- `--plot-psi` が指定され、データに `alpha` がある場合：
  - true: `psi_true = lam ** alpha`
  - learned: `psi_pred = psi(lam)`
  - `plot_psi_curve` で比較（log-log推奨）

---

## 5. `gen_fractional_diffusion_1d.py` の詳細仕様（実装要件）

### 5.1 CLI 引数（最低限）
- `--out-file`（.mat 出力先）
- `--N`（サンプル数）
- `--S`（空間グリッド）
- `--T`（時刻数）
- `--t-max`
- `--alpha`（分数拡散指数）
- `--seed`
- `--grf-beta`（初期条件の滑らかさ。スペクトル減衰指数）
- `--grf-scale`（振幅）

### 5.2 初期条件 a の生成（GRF）
- 周期 GRF をスペクトル合成で生成（rfft形式で係数を作る）
- 周波数：`k = rfftfreq(S, d=1/S)` → 0..S/2
- 角周波数：`w = 2πk`
- `lam = w^2`
- 複素ガウス：`z = (N(0,1)+iN(0,1))/sqrt(2)`
- 振幅：`amp = grf_scale * (1 + lam) ** (-grf_beta/2)`
- `a_hat = amp * z`（k=0 成分は 0 にして平均0でもよい）
- `a = irfft(a_hat, n=S)` → real
- `a` を各サンプルで標準化（mean 0 / std 1）してもよい（推奨）

### 5.3 解 u の生成（FFTで厳密）
- 時刻：`t = linspace(0, t_max, T)`
- 真の ψ：`psi_true(lam) = lam ** alpha`
- 解：
  - `mult = exp(-t[:,None] * psi_true[None,:])` → shape `(T,K)`
  - `u_hat(t) = a_hat[None,:] * mult` → `(T,K)`
  - `u(t) = irfft(u_hat, n=S)` → `(T,S)`
  - 転置して `(S,T)` にし、`u[i]` として保存

### 5.4 出力 .mat
- `a`: `(N,S)` float32
- `u`: `(N,S,T)` float32
- `t`: `(T,)` float32
- `alpha`: scalar
- 追加メタ：`S`, `T`, `t_max`, `grf_beta`, `grf_scale`

---

## 6. README 追記内容（最低限の例）
### データ生成
```bash
python data_generation/fractional_diffusion_1d/gen_fractional_diffusion_1d.py \
  --out-file data/fractional_diffusion_1d_alpha0.5.mat \
  --N 1200 --S 1024 --T 11 --t-max 1.0 --alpha 0.5 --seed 0
学習
python subordination_1d_time.py \
  --data-mode single_split --data-file data/fractional_diffusion_1d_alpha0.5.mat \
  --ntrain 1000 --ntest 200 --sub 2 --sub-t 1 \
  --batch-size 20 --learning-rate 1e-2 --epochs 300 \
  --psi-J 32 --learn-s --psi-s-min 1e-3 --psi-s-max 1e3 \
  --plot-psi
7. Smoke Test（完成判定の最低条件）

Codex 実装後、以下が通ること：

データ生成が成功し .mat が作られる

subordination_1d_time.py がエラーなく学習を開始し、少なくとも数epochで loss が下降傾向

visualizations/subordination_1d_time/ に

learning_curve（png/pdf/svg）

error_hist（png/pdf/svg）

sample plots（png/pdf/svg）

（plot-psi指定時）psi_curve（png/pdf/svg）
が生成される

既存スクリプト（fourier_1d.py など）を壊さない

8. 実装上の注意（バグ防止の要点）

周期前提。非周期（Dirichlet等）には未対応（この段階では対応不要）。

S はデータから取得し、sub 後のサイズで lam を作る。

torch.fft.rfftfreq(S, d=1/S) は周波数が整数列になる（n*d=1）ので扱いやすい。

exp(-t*psi) は underflow して 0 になっても OK（安定）。NaN だけ防ぐ。

device は cuda があれば使うが、CPUでも動くよう .to(device) を推奨。

既存の repo 文化に合わせ、過度な抽象化より「読みやすい stand-alone スクリプト」を優先。

9. Definition of Done（納品物）

subordination_1d_time.py（新規）

data_generation/fractional_diffusion_1d/gen_fractional_diffusion_1d.py（新規）

viz_utils.py に plot_psi_curve を追加（推奨）

README.md に実行例を追加

# AGENT.md — Subordination 1D: 修正点 + (A) ψ解析的ベースライン推定 + (B) Monte Carlo subordination forward

## 0. このリポジトリの現状（前提）
この repo には既に以下が実装済み：
- `subordination_1d_time.py`
  - 1D 周期領域 `[0,1)` の半群
    \[
      \widehat{u}(k,t)=e^{-t\psi(\lambda_k)}\widehat{a}(k),\quad \lambda_k=(2\pi k)^2
    \]
    を FFT で実装し、Bernstein 関数 `ψ`（離散 Lévy measure 近似）を学習する。
- `data_generation/fractional_diffusion_1d/gen_fractional_diffusion_1d.py`
  - 分数拡散（真の \(\psi(\lambda)=\lambda^\alpha\)）のデータ生成
- `viz_utils.py`
  - `plot_psi_curve` 等の可視化ユーティリティ

学習・推論は動作しているが、コードレビューで見つかった修正点と、追加で以下 2 点を実装したい：

(A) **ψ の解析的ベースライン推定（log-ratio）**  
(B) **Monte Carlo subordination forward（「熱方程式を何回も回す」形）**

本 AGENT.md は Codex CLI が会話履歴無しで実装できるよう、必要仕様を全て記述する。

---

## 1. 修正点（バグ/再現性/README整合）

### 1.1 `BernsteinPsi` の `s_j` 初期化が意図とズレている（重要）
現状 `BernsteinPsi` は
- `s0 = logspace(s_min, s_max)`
- `log_s = log(s0 - eps)`
- `s = softplus(log_s) + eps`
の形で `s_j>0` を保証しているが、この変換は一般に **`s != s0`** となり、特に `s0` が大きいと大幅に縮む（例：s0=1000 → s≈6.9）。  
固定 `s` の場合、表現力が設計通りにならないため修正する。

#### 修正方針（必須）
`softplus` ではなく **指数パラメータ化**に置き換える：

- `theta_s` を unconstrained パラメータとして保持し、
  \[
    s_j = \exp(\theta_{s,j}) + \varepsilon_s
  \]
- 初期化は
  \[
    \theta_{s,j} = \log(s_{0,j})
  \]
  とし、**初期値が厳密に logspace に一致**するようにする。
- `--learn-s` が無い場合は `theta_s` を `register_buffer` で固定する（現行と同じ挙動）。

#### ついでに推奨（任意だが入れるのが望ましい）
- `a,b` は `psi(0)=a` で平均モードを減衰させうるので、初期値を小さくすると安定：
  - `log_a` / `log_b` の初期値を `-10.0` などにして `softplus` 後ほぼ 0 から開始する。

### 1.2 `subordination_1d_time.py` の x グリッド生成（軽微）
周期 `[0,1)` なら描画用 `x` は `endpoint=False` が自然：
- 現状：`np.linspace(0,1,S)`
- 修正：`np.linspace(0,1,S, endpoint=False)`

### 1.3 README 実行例の整合性（必須）
現状 README の例：
- 生成：`N=1200`
- 学習：`ntrain=1000, ntest=200`
- しかし `subordination_1d_time.py` の default `--train-split 0.8` だと train=960 < 1000 で失敗。

#### 修正方針（必須）
README の **データ生成**例を `--N 1500`（またはそれ以上）に変更し、
`--train-split` を指定せずとも `ntrain=1000, ntest=200` が満たされるようにする。

---

## 2. (A) ψ の解析的ベースライン推定（log-ratio）を実装

### 2.1 追加ファイル（必須）
- `scripts/estimate_psi_logratio_1d.py`（新規、stand-alone）

### 2.2 数式（この推定器の定義）
モデルクラス（subordination で表現できる線形・定係数・周期系）では
\[
\widehat{u}(k,t)=e^{-t\psi(\lambda_k)}\widehat{a}(k)
\]
したがって絶対値を取って
\[
\log|\widehat{u}(k,t)|-\log|\widehat{a}(k)| = -t\,\psi(\lambda_k)
\]
より、各モード \(k\) で原点回帰（intercept 0）の最小二乗：
\[
\hat\psi(\lambda_k)
=
-\frac{\sum_{i,t} w_{i,k,t}\, t\; y_{i,k,t}}{\sum_{i,t} w_{i,k,t}\, t^2},
\quad
y_{i,k,t}=\log(|\widehat{u_i}(k,t)|+\epsilon)-\log(|\widehat{a_i}(k)|+\epsilon)
\]
ここで \(w\) は数値安定のためのマスク（閾値）で良い。

### 2.3 実装仕様（必須）
#### 入力データ
- `.mat` に必須フィールド：
  - `a`: `(N,S)`
  - `u`: `(N,S,T)`  （time-last）
  - `t`: `(T,)`
- 任意：`alpha`（分数拡散の真値比較用）

#### FFT
- `a_hat = torch.fft.rfft(a, dim=1)` → `(N,K)` complex, `K=S//2+1`
- `u_hat = torch.fft.rfft(u, dim=1)` → `(N,K,T)` complex

#### ベースライン推定
- `abs_a = |a_hat|`, `abs_u = |u_hat|`
- `y = log(abs_u + eps) - log(abs_a[...,None] + eps)`  shape `(N,K,T)`
- マスク：
  - `mask = (abs_a > amp_thr)[...,None] & (abs_u > amp_thr)`
- `num = sum_{i,t} (mask * t * y)` 、 `den = sum_{i,t} (mask * t^2)`
- `psi_hat = clamp(-num/(den+tiny), min=0)` （負値は 0 に丸めても良い）
- `k=0`（λ=0）は `psi_hat[0]=0` を明示（den が 0 の場合があるため）

#### 出力
- `visualizations/.../psi_baseline_curve.(png/pdf/svg)`（`viz_utils.plot_psi_curve` を使用）
- 可能なら `npz` 保存（再利用・解析用）：
  - `lam`, `psi_hat`, `counts`（mask の有効数）, `t`, `alpha(optional)`

### 2.4 CLI 仕様（最低限）
`cli_utils.py` を使い既存スクリプトと同じスタイルにする。

- データ:
  - `--data-mode {single_split,separate_files}`
  - `--data-file` / `--train-file` / `--test-file`
  - `--sub`, `--sub-t`
  - `--seed`, `--shuffle`, `--train-split`
  - `--split {all,train,test}`  
    - `single_split`: train/test は `train_split` を使って分割（`shuffle`/`seed` も反映）
    - `separate_files`: train/test は該当ファイル、all は concat

- 推定:
  - `--amp-threshold`（例：1e-8）
  - `--log-eps`（例：1e-12）
  - `--max-samples`（0 or negative で無制限。大規模データでの高速化用）

- 出力:
  - `--viz-dir`（例：`visualizations/psi_baseline_1d`）
  - `--out-npz`（デフォルト：`<viz-dir>/psi_baseline.npz`）
  - `--plot-psi-true`（alpha があれば true ψ=λ^alpha を重ねる）

---

## 3. (B) Monte Carlo subordination forward を実装（熱方程式を何回も回す）

### 3.1 目的
現在の forward は直接
\[
e^{-t\psi(\lambda)}
\]
を掛けているが、subordination の本質は
\[
S(t)a=\mathbb{E}[T(\tau)a]\;e^{-a t}
\]
という「**熱半群 \(T\) をランダム時間 \(\tau\) で何度も回して平均**」にある。  
これを実装して、deterministic multiplier と一致することを検証できるようにする。

### 3.2 離散 Bernstein 表現に対する確率過程（実装で使う）
Bernstein 近似：
\[
\psi(\lambda)=a + b\lambda + \sum_{j=1}^J \alpha_j(1-e^{-s_j\lambda})
\]
に対し、独立な Poisson 過程 \(N_j(t)\sim \mathrm{Poisson}(\alpha_j t)\) を使って
\[
\tau(t)= b t + \sum_{j=1}^J s_j N_j(t)
\]
とすると
\[
\mathbb{E}[e^{-\lambda\tau(t)}]=\exp\left(-t\left(b\lambda + \sum_j \alpha_j(1-e^{-s_j\lambda})\right)\right)
\]
ゆえに
\[
e^{-t\psi(\lambda)} = e^{-a t}\;\mathbb{E}[e^{-\lambda\tau(t)}].
\]

熱半群（周期、フーリエ対角）：
\[
\widehat{T(\tau)a}(k)=e^{-\tau\lambda_k}\widehat{a}(k)
\]
なので
\[
S(t)a=e^{-a t}\,\mathbb{E}[T(\tau(t))a].
\]

### 3.3 実装仕様（必須）
#### 変更/追加：`BernsteinPsi`
- **`positive_params()`** もしくは同等の getter を追加：
  - `a0 = softplus(log_a)`（scalar）
  - `b = softplus(log_b)`（scalar）
  - `alpha = softplus(log_alpha)`（shape `(J,)`）
  - `s = exp(theta_s) + s_eps`（shape `(J,)`）  ※1.1の修正後
- これにより MC forward が `psi(lam)` を再計算せずにパラメータへアクセスできる。

#### 変更/追加：`SubordinatedHeatOperator1D`
- **`forward_mc(a, t, mc_samples, generator=None, chunk_size=None)`** を追加（`forward` は従来通り deterministic のまま）
- 入出力：
  - `a`: `(B,S)` または `(B,S,1)`
  - `t`: `(T,)`
  - 出力：`(B,S,T)`

- 実装（推奨：T 方向にループ）
  1) `a_hat = rfft(a)` → `(B,K)` complex  
  2) `lam = self.lam` → `(K,)`
  3) パラメータ取得：`a0,b,alpha,s`
  4) 時刻 `t_j` ごとに：
     - `rates_j = alpha * t_j`（shape `(J,)`）
     - `N ~ Poisson(rates_j)` を **(mc_samples, B, J)** でサンプル  
       - `torch.poisson(rate_tensor, generator=generator)` を使う
       - rate_tensor は `rates_j[None,None,:].expand(mc_samples, B, J)` のように作る
     - `tau = b*t_j + sum_j s_j * N_j` → `(mc_samples, B)`
     - 熱半群：`mult = exp(-tau[...,None] * lam[None,None,:])` → `(mc_samples,B,K)`
     - `u_hat_samples = a_hat[None,:,:] * mult` → `(mc_samples,B,K)`
     - `u_hat_mean = mean(u_hat_samples, dim=0)` → `(B,K)`
     - killing：`u_hat_mean *= exp(-a0*t_j)`
     - `u_j = irfft(u_hat_mean, n=S)` → `(B,S)`
     - 出力テンソルの `[:, :, j]` に格納

- メモリ対策（推奨）：
  - `chunk_size` が指定されたら MC サンプル軸を分割して平均を逐次加算する（大きい `mc_samples` で OOM しないように）

### 3.4 `subordination_1d_time.py` への統合（必須）
学習は deterministic のままでよい（MC は微分が重いので評価専用）。

#### CLI 追加
- `--mc-samples`（int, default=0：0ならMC評価しない）
- `--mc-seed`（int, default=0）
- `--mc-batch-size`（int, default=`batch_size`）
- `--mc-chunk`（int, default=0：0なら chunk 無し）

#### 実行内容（学習後の評価パート）
- 既存の deterministic `pred_test` を計算した後、`mc_samples>0` なら：
  - DataLoader（batch_size=`mc_batch_size`）で `x_test` を回し、
    `model.forward_mc(a_batch, t_dev, mc_samples, generator=..., chunk_size=...)` で `pred_test_mc` を作る
  - `mc_seed` で generator を固定（CPU/GPU どちらでも再現できるようにする）

#### 追加で作る評価・可視化（必須）
- per-sample で
  - `err_mc_vs_det[i] = rel_l2(pred_test_mc[i], pred_test_det[i])`
  - `err_mc_vs_gt[i]  = rel_l2(pred_test_mc[i], y_test[i])`
- ヒストグラムを保存：
  - `mc_vs_det_hist.*`
  - `mc_vs_gt_hist.*`（任意だが推奨）
- 代表サンプル/代表時刻（t=0, T//2, T-1）で
  - GT / deterministic / MC を同じ図に描画して保存（次項の viz_utils 拡張を使う）

---

## 4. viz_utils 拡張（推奨だが入れること）
### 4.1 追加関数：`plot_1d_prediction_multi`（新規）
目的：GT に対して複数の予測（deterministic と MC）を重ねて可視化。

提案シグネチャ：
```python
def plot_1d_prediction_multi(
    x: Optional[ArrayLike],
    gt: ArrayLike,
    preds: dict[str, ArrayLike],
    out_path_no_ext: str,
    input_u0: Optional[ArrayLike] = None,
    title_prefix: str = "",
) -> Tuple[str, str, str]:
    ...
```

仕様：

preds の各系列をラベル付きで描画（matplotlib デフォルトの色サイクルに任せる）

各 pred の relL2 と RMSE を計算し、title に含める（複数行でもOK）

png/pdf/svg で保存（既存 save_figure_all_formats を使用）

subordination_1d_time.py 側で plot_1d_prediction をこれに置き換えるか、MC評価時のみ multi を使う。

5. README 更新（必須）
5.1 既存の N 不整合を修正

データ生成例の --N 1200 を --N 1500 に修正（もしくはそれ以上）

5.2 (A) ベースライン推定スクリプトの使い方を追記

例：

python scripts/estimate_psi_logratio_1d.py \
  --data-mode single_split --data-file data/fractional_diffusion_1d_alpha0.5.mat \
  --sub 2 --sub-t 1 --split all \
  --viz-dir visualizations/psi_baseline_1d \
  --plot-psi-true
5.3 (B) MC 評価の使い方を追記

例：

python subordination_1d_time.py \
  --data-mode single_split --data-file data/fractional_diffusion_1d_alpha0.5.mat \
  --ntrain 1000 --ntest 200 --sub 2 --sub-t 1 \
  --batch-size 20 --learning-rate 1e-2 --epochs 300 \
  --psi-J 32 --learn-s --psi-s-min 1e-3 --psi-s-max 1e3 \
  --plot-psi \
  --mc-samples 256 --mc-seed 0 --mc-batch-size 20 --mc-chunk 0
6. Smoke Test（完了判定）

最低限、以下が通ること：

データ生成（README例）が成功し .mat が作成される

subordination_1d_time.py が学習でき、図が保存される（従来機能が壊れていない）

scripts/estimate_psi_logratio_1d.py が動作し、psi_baseline_curve.* が保存される

--mc-samples 128 以上で MC を回せ、mc_vs_det_hist.* が生成される

mc_vs_det は概ね小さくなる（サンプル数を増やすと改善する傾向が出る）

変更によって他スクリプト（fourier_1d.py 等）が壊れない（依存追加なし）

7. 実装上の注意

repo の文化：stand-alone スクリプトを優先し、過度な抽象化はしない。

新規依存追加は禁止（既存：numpy/torch/matplotlib/scipy.io などの範囲で完結）。

GPU/CPU 両対応（device に乗る tensor で torch.poisson できるように rate_tensor も device を揃える）。

数値安定：

log は log(x+eps) を徹底

underflow は 0 になっても良いが NaN を出さない
