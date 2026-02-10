# AGENT.md — Add ExtremONet to POL_for_PDE and make FNO/EON cross-compatible

## ミッション（必達）
POL_for_PDE リポジトリに ExtremONet(EON) を実装し、既存の FNO スクリプトと同様に
- CLI から学習・評価・可視化・保存ができる
- FNO用データ(.mat) と EON用データ(.pkl) の **両方** を
  - FNO でも実行できる
  - EON でも実行できる
ようにする。

### 絶対条件
1. 既存スクリプト（fourier_1d.py / fourier_2d.py / fourier_2d_time.py / fourier_3d.py / scripts/*）の既存挙動を壊さない。
2. 新規機能は「追加 or オプション」で提供する（デフォルトは従来通り動く）。
3. EONモデルは **保存→ロード→推論** が再現できる（EONのランダム特徴も state_dict に入るようにする）。
4. データ変換で巨大メモリを使い切らない設計（必要ならサンプリング/分割/ストリーミング集計を入れる）。

---

## 前提：ExtremONet-MLDE を codex-cli から参照できる位置に置く

### 推奨（vendor方式）
この POL_for_PDE リポジトリ直下に `external/ExtremONet-MLDE-main/` を置く。

例（ユーザが手元で実行）:
```bash
mkdir -p external
# ExtremONet-MLDE-main を external にコピー（zip展開済み想定）
cp -r /path/to/ExtremONet-MLDE-main external/ExtremONet-MLDE-main
```
