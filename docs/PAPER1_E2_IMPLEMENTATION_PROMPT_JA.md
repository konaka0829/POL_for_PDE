# 第一論文 E2 実装仕様

E2は、E0で検証されたBurgers master datasetを用い、同じ有限入力
`n_ref -> n_tar -> n_sur`、sample ID、train/validation/test分割、固定J観測で
Model 1--3を比較する。E1出力はruntime dependencyにしない。

必須protocol:

1. Burgersと反応拡散について、固定初期時刻で粘性係数をvalidation sweepする。
2. config指定representative modelのvalidation誤差で粘性を選び、その粘性で
   読み出し時刻をvalidation sweepする。
3. model-specific optimumとshared representativeを別artifactへ保存する。
4. selection recordを保存・hash確定した後にのみtest curveを評価する。
5. Model 2はcentered affine ridgeを用い、zetaはvalidation、tieはlargest zeta。
6. Model 3は `concat(phi, rho(phi A^T+c)/sqrt(M))`、skip connection、seed平均選択、
   selection/evaluation seed分離を用いる。
7. Model 1でq>Jの場合、観測格子で一意な通常低波数係数だけをコピーし、
   unavailable modeをzero paddingする。
8. shared point選択後、共通格子terminal field、fixed-J feature、finestで固定した
   readout predictionのn_sur収束を確認し、family baseとglobal maximumをE3へ渡す。
9. state cacheとfeature cacheを分離し、model/q/zeta/Model 3 seedをstate keyへ入れない。
10. JSONはNaN禁止、CSV/PT/JSONをread-backし、欠損・重複・非有限・hash改変を拒否する。

主成果物は `selection_record.json`、validation/test CSV、Model 3 seed統計、
`model_specific_optima.json`、`shared_representatives.json`、convergence artifacts、
pass時のみの `e2_handoff.json`、plots、summary、artifact manifestである。

Productionの4096×1400 sweepは本実装作業では実行せず、smoke configと同じcode
pathによるE0→dataset→E2、resume、改変拒否を検証する。
