# Slide Asset Generation

This workflow creates one figure component per file for PowerPoint editing. It does not create Model 2 / Model 3 prediction space-time plots. Use:

- target Burgers space-time: `u(x,t)`
- surrogate PDE space-time: `utilde(x,tau)`
- final-time comparison: `y(x,T)` vs `yhat(x,T)`

## B0 Smoke

```bash
python scripts/generate_burgers_1d.py \
  --config configs/B0_smoke.json \
  --format pt \
  --out-file data/burgers_B0_smoke.pt \
  --device cpu
```

```bash
python model123_burgers_1d.py \
  --config configs/B0_smoke.json \
  --data-mode single_split \
  --data-file data/burgers_B0_smoke.pt \
  --model model2 \
  --reservoir reaction_diffusion \
  --rd-nu 0.001 \
  --rd-alpha 1.0 \
  --rd-beta 1.0 \
  --T 0.1 \
  --Ttilde 0.1 \
  --dt 0.01 \
  --K 1 \
  --obs full \
  --ridge-zeta 1e-8 \
  --ntrain 12 \
  --nval 6 \
  --ntest 6 \
  --data-dtype preserve \
  --sim-dtype float64 \
  --ridge-dtype float64 \
  --device cpu \
  --out-dir outputs/smoke_slide/model2_rd \
  --save-model \
  --save-predictions
```

```bash
python scripts/make_model123_stage_waveform_assets.py \
  --run-dir outputs/smoke_slide/model2_rd \
  --sample-index 0 \
  --out-dir outputs/smoke_slide/assets/model2_rd \
  --shared-ylim
```

```bash
python scripts/make_slide_spacetime_assets.py \
  --config configs/B0_smoke.json \
  --predictions outputs/smoke_slide/model2_rd/predictions.pt \
  --run-dir outputs/smoke_slide/model2_rd \
  --trajectory both \
  --sample-index 0 \
  --num-frames 21 \
  --out-dir outputs/smoke_slide/assets/spacetime \
  --device cpu \
  --dtype float64
```

## B1 Production Example

`configs/B1_burgers_grf.json` uses target viscosity `nu = 1e-2`. Match this value in the slide text and formulas.

Generate data:

```bash
python scripts/generate_burgers_1d.py \
  --config configs/B1_burgers_grf.json \
  --format pt \
  --out-file data/burgers_B1_grf.pt \
  --device cuda \
  --batch-size 20
```

Run the reaction-diffusion sweep:

```bash
python scripts/run_model123_param_sweep.py \
  --config configs/B1_burgers_grf.json \
  --data-file data/burgers_B1_grf.pt \
  --out-root outputs/final/rd_nu_alpha_grid \
  --models model1,model2,model3 \
  --reservoir reaction_diffusion \
  --sweep rd_nu=0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1 \
  --sweep alpha=0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 \
  --rd-alpha 1.0 \
  --rd-beta 1.0 \
  --T 1.0 \
  --dt 0.001 \
  --K 1 \
  --obs full \
  --ridge-zeta 1e-8 \
  --ntrain 800 \
  --nval 200 \
  --ntest 200 \
  --data-dtype preserve \
  --sim-dtype float64 \
  --ridge-dtype float64 \
  --burgers-scheme etdrk4 \
  --burgers-dealias 1 \
  --batch-size 20 \
  --device cuda \
  --max-workers 3
```

Rerun the best runs with saved model states and predictions:

```bash
python scripts/rerun_best_model123_for_slide_assets.py \
  --sweep-root outputs/final/rd_nu_alpha_grid \
  --out-root outputs/slide_runs/rd_best \
  --rank 0 \
  --device cuda
```

Create stage waveform assets:

```bash
python scripts/make_model123_stage_waveform_assets.py \
  --run-dir outputs/slide_runs/rd_best/model1 \
  --sample-index 0 \
  --out-dir outputs/slide_assets/rd_best/model1 \
  --shared-ylim

python scripts/make_model123_stage_waveform_assets.py \
  --run-dir outputs/slide_runs/rd_best/model2 \
  --sample-index 0 \
  --out-dir outputs/slide_assets/rd_best/model2 \
  --shared-ylim

python scripts/make_model123_stage_waveform_assets.py \
  --run-dir outputs/slide_runs/rd_best/model3 \
  --sample-index 0 \
  --out-dir outputs/slide_assets/rd_best/model3 \
  --shared-ylim
```

Create space-time assets. The target Burgers trajectory is identical for all models, so reuse `model1_sample000_target_burgers_spacetime.*` created by the first command.

```bash
python scripts/make_slide_spacetime_assets.py \
  --config configs/B1_burgers_grf.json \
  --predictions outputs/slide_runs/rd_best/model1/predictions.pt \
  --run-dir outputs/slide_runs/rd_best/model1 \
  --trajectory both \
  --sample-index 0 \
  --num-frames 101 \
  --out-dir outputs/slide_assets/rd_best/model1_spacetime \
  --device cuda \
  --dtype float64 \
  --symmetric-colorlim

python scripts/make_slide_spacetime_assets.py \
  --config configs/B1_burgers_grf.json \
  --predictions outputs/slide_runs/rd_best/model2/predictions.pt \
  --run-dir outputs/slide_runs/rd_best/model2 \
  --trajectory surrogate \
  --sample-index 0 \
  --num-frames 101 \
  --out-dir outputs/slide_assets/rd_best/model2_spacetime \
  --device cuda \
  --dtype float64 \
  --symmetric-colorlim

python scripts/make_slide_spacetime_assets.py \
  --config configs/B1_burgers_grf.json \
  --predictions outputs/slide_runs/rd_best/model3/predictions.pt \
  --run-dir outputs/slide_runs/rd_best/model3 \
  --trajectory surrogate \
  --sample-index 0 \
  --num-frames 101 \
  --out-dir outputs/slide_assets/rd_best/model3_spacetime \
  --device cuda \
  --dtype float64 \
  --symmetric-colorlim
```
