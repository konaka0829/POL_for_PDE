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
