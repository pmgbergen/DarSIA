# FF_AC round 5: the AC17 sand map in every run, refit, production

Specification, 20 Oct 2026. It runs after the code lock.

**Status: proposal.**
- Nothing in it runs before the decider has answered the decisions in section 0.
- Nothing runs before the lock except step A1. A1 only writes a new review folder and reads nothing that the lock gates use.

**Why.**
- The decider, 19 Oct: the phase model should not react to a new sand map; the calibration must be updated.
- Known weakness #2 (14 Oct): the old facies map's layer boundaries sit 5–16 mm above the visible layer edges. Depth, porosity and colour path were therefore taken from the wrong layer in those bands.
- Weakness #1 (14 Oct): there is one common calibration, which is refitted once after the facies-map fix.
- The decider is redrawing the sand map on AC17 in DarSIA's painting tab (N83). Once it is approved, it replaces the AC14 map as the source. It is placed in every run through that run's own confirmed frame model, as candidate A was. There is no per-run settling adaptation (N53).
- Round 4 (12–13 Oct) is the precedent: a corrected sand map, a refit of a and β, and the same phase maps. Its spec is `SPEC_ffac_round4_edge_sand_refit_production_20261012_v2.md`, and its score is in `scores/ffac-round4-score-20261013.md`.

## 0. Decisions for the decider

The recommended choice for each is marked.

1. **Approve the AC17 sand map.**
   - The session is `X:\FF_AC\DarSIA\geometry_facies\annotations\layer_painting\ac17.json`, last saved 19 Oct, 22:00 Oslo.
   - His approval names the SHA-256 of the approved file.
2. **Mass geometry in this round.**
   - **(a) Today's geometry:** 889 mm width, no depth offset, no bolt perspective.
   - **(b) The measured geometry (recommended):**
     - 899 mm between the B edges (N42);
     - the B-to-sand depth offset of about 14 mm (6 mm frame plus 12 mm plate divided by about 1.5);
     - the option-B horizontal perspective from the bottom bolts (Q86-B, B confirmed 39/39 in N39b).

   Lengths change by about +2.8 % and areas by about +5.7 %. The gain absorbs part of this, but free gas does not.

   Option (b) is adopted only after the decider has seen the option-B frame figures, as Q86-B requires.

   With (b), one refit covers both the map and the geometry. With (a), a later adoption of (b) needs one more refit.
3. **Colour paths.**
   - **Rebuilt with the new map (B1, recommended).** The round-1 paths that rounds 2–4 reused were built from pixels labelled with the old, offset boundaries.
   - **Kept.**
4. **Overflow-governed free gas (step F).**
   - **Prepare and evaluate it in this round as a separate variant (recommended).** The decider decides on its use after seeing the numbers.
   - **Later.**
5. **Production set.**
   - **The 80-image job:** about 3,090 frames on grid v2. It needs its own GO.
   - **Today's selection:** 1,603 frames, that is 1,564 selected plus 39 baselines.

   The calibration steps are the same in both cases.

## 1. Unchanged from round 4

- **Phase model and phase maps.**
  - The phase model is P1 final 3, inference only. It uses the frozen gas envelopes E1 and E2 and history rule A: every selected image builds the history in time order, and each run is computed from start to end on one backend.
  - **Old map pinned.** The v5 light normalisation and the phase step's `target_context` keep the old facies map. Then the phase maps cannot change because of the new map.
  - Re-deriving v5's per-facies levels on the new map (weakness #10) is **not** part of this round. It would change the corrected images and therefore the phase maps, and it needs its own decision.
- **Runs.**
  - 26 fit runs.
  - 12 held-out runs: AC19, AC25, AC35, AC41, AC47 (rig light); AC52, AC56, AC60 (new lamps); AC21, AC22, AC32, AC44 (caveat).
  - AC38 is out of the fit and descriptive.
  - The chemical outliers AC20, AC24, AC27 and AC51 are left out.
- **Doses.** The measured MFC integral where the log passes QA. Nominal doses for AC17, AC31, AC45 and AC57–61, flagged. AC38 nominal.
- **Recipes.** Each run's recipe is a measured input to the titration flash.
- **Model N1:** `TitrationFlash` with each run's recipe, times one common gain a, clipped to 0–1.
- **Dilution law.** f = min(1, (V/V_end)^−β), applied only to aqueous concentration and fitted in stage 2.
- **Gas rule.** In label 2, s_g = 1 − Swi of the pixel's facies, from `data\facies.csv`. Nothing is fitted in the gas support.
- **Interval and width.** [M + E_lo, M + E_hi + U_rest] and W = E_hi − E_lo + U_rest (round 3, §2).
- **Main mass M.** As defined by the locked code's domain rule. The report states which definition is used.
- **Overflow** is compare only in the fit.
- **No rescaling.** No map is rescaled to the dose.
- **Precision:** float32 for everything produced (policy v2, `4f73dc13…`). Equivalence limits follow addendum 8 v3 (`36aa8e5f…`).
- **Roles.** One process computes and another scores once. Production waits for the decider's GO.

## 2. Steps

### A. The AC17 map in every run

**A1 (may run before the lock, read-only for every gate input).**

- **Placement.** Place the approved AC17 map in all 39 runs through each run's own confirmed frame model, with the existing importer `geometry_import`. Write only to a new review folder. Apply no settling adaptation (N53).
- **Overlays.** Make colour overlays of the map on each run's own photo:
  - the whole image, at full resolution, with free zoom;
  - at most two labelled line types per panel;
  - English text.

  Show AC17 first as the control, then AC28, AC47 and AC56 (one run per period), then the rest, with an index.
- **Offsets (descriptive).** Measure each layer's offset per run with the existing method: a ±25 mm band, the median, and the 10th and 90th percentiles.
- **Checks to report:**
  - every facies id in the map exists in `facies.csv` and in the colour-path table;
  - pixels labelled water inside the sand domain. N83 found AC14's labels 1 and 2 labelled water where AC17 shows coarse sand;
  - gaps that need the round-4 fill: nearest sand in the same row, then in the same column.
- **The decider looks at the overlays and approves before A2.**

**A2 (after the lock).** Adopt the map as a new workspace revision with logged adoption. Then run "Apply geometry to Setup" and "Build run model" into new folders. Re-point the run configurations with a logged migration. The gate is that a re-export reproduces the adopted per-run labels exactly. With decision 2(b), the per-frame pixel-area maps come from the measured frame geometry.

### B. Production phase 1: phase maps

- **Code.** The locked code with RAW in the ordinary job queue (N155, N156).
- **Run order:** the 26 fit runs first, then the 12 held-out runs, then AC38. Step C can then start run by run while phase 1 continues.
- **Machines.** Both machines, whole runs per worker. The first computer runs CPU and CUDA. The second computer runs CPU, and OpenCL if it is qualified. No machine may become the bottleneck.
- **Stored per frame:**
  - the phase labels and statistics;
  - the per-pixel gas probability, both pixel-head and regional (OVF GO, 18 Oct).
- **Checks after phase 1**, as on 10 Oct:
  - the bound at the Studio times;
  - the sealed scorers: recall drop ≤ 0.03 and false-gas rise ≤ 0.01;
  - the time series of CO2 and gas area.

### C. Fit extraction and refit

The fit uses the grid-v1 times, which are a subset of grid v2, so that round 5 compares directly with round 4.

- **B0, the baseline.** Round 4's frozen calibration `bce49292…` (N1, a 23.859, β 0.6513) with its colour paths, applied unchanged with the new map, and with the new geometry if decision 2(b) is taken.
- **B1, the refit:**
  1. **Colour paths.** One path per facies, pooled over the 26 fit runs, relative to each run's own baseline. Use the round-1 method from the common-calibration spec, §3 item 2. Facies without enough data stay an unquantified bound.
  2. **Stage 1, the gain a.** Use the selected injection images from 0.5 h (0.500, 0.625, …, 2.375 h; 16 times), with the two-sided residual on the interval.
  3. **Stage 2, β.** Use every selected post-injection image from 2.5 h to the last, where W ≤ 0.05·d. Each run has equal weight, split equally between the bins 2.5–8 h, 8–48 h and after 48 h. The gain stays fixed.
  4. **Optimizer check.** Run a grid over a and a grid over β for B1 and its folds. If a grid finds a lower objective, report it and stop.
  5. **LORO.** 26 folds, about 30 minutes each, eight at a time. They give the fit-run predictions for check 3.
- **Freeze B1** (files and SHA-256) before any held-out frame is computed with it.

### D. Held-out frames and checks (one process computes, another scores once)

The checks are those of round 4 (amendment 1), fixed before any result:

1. **Check 1.** The median |M/d − 1| over the 12 held-out runs is ≤ 0.10 at 1.0 h and at 2.25 h, and each group median lies within 0.90–1.10.
2. **Check 2**, at 4, 8, 12, 24 and 48 h on the held-out runs.
   - δ = 0 inside the interval; otherwise it is the relative distance to the nearer end.
   - A run counts at a time if U_rest is defined and W ≤ 0.05·d, and a time needs at least 7 counted runs.
   - Pass if the median δ ≤ 0.10 at every assessable time and at least 4 of the 5 times are assessable.
3. **Check 3**, at 56, 64, 72 and 96 h. It uses the held-out runs plus the fit runs, each predicted by its LORO fold. The same rule applies, with at least 3 of the 4 times assessable.
4. **Fallback (amendment 1).** Only if check 2 has fewer than 4 assessable times, or check 3 fewer than 3, is that check scored with U_rest = 0 (`40c503b7…`). That score then decides. An assessable check that fails, fails. Both tables are reported.

**Decision:**
- If B1 passes, B1 is the calibration.
- If B1 fails and B0 passes, the decider decides.
- If both fail, stop and report where the calibration holds and where it fails.

**Also reported (descriptive):**
- **Period difference (weakness #3).** The median M/d at 24–48 h for AC17–38, AC39–49 and AC50–61, under B0 and B1. Round 4 had 0.99, 1.02–1.03 and 1.06.
- **Changes against round 4,** per run and time:
  - the masses M, E_lo, E_hi and U_rest;
  - the gas mass and its share of the total.
- **The colour paths,** B1 against round 4, per facies.

### E. Production phase 2: mass products

This step runs after the decider's GO, and only if check D passes.

- **For all 39 runs:**
  - the mass products: per-pixel mass for aqueous and free gas (N44b), concentration, gas saturation, labels and masks;
  - the mass tables;
  - films at 3 fps;
  - figures in English.
- **Storage.** The job uses the profile and the stored resolution the decider chooses in New analysis. Recommended: Lean with the 1 mm layer written in the run (N46, N55).
- **Order.** AC19, AC28, AC47, AC52 and AC56 first. Wasserstein waits.

### F. Variant: overflow-governed free gas

This is a proposal. It runs only after decision 4, and its use in production is the decider's decision after the numbers.

**Direction (the decider, 18 Oct):**
- free gas is reduced to match overflow;
- the reduction follows gas probability;
- the level is rooted in each run's own overflow.

The phase maps stay unchanged.

1. **Overflow per run.**
   - Use the recorded overflow review: feed model C, with the anchor set after the pre-injection drop. That drop is the pH sampling, about 20 mL, and the anchor is set per run.
   - The source is `ffac_overflow_review_20261018\results\overflow_review.json` (`f39dd556…a8ee`).
   - 24 runs are usable. The other runs, and runs without data, get the group value.
2. **Per-run level.**
   - R_r is the median, over the selected frames in 1.0–4.0 h, of the ratio of overflow gas volume to model free-gas volume under B1.
   - Shrink it toward the group value: k_r = w_r·R_r + (1 − w_r)·R_group, with w_r = τ²/(τ² + σ_r²).
     - σ_r is the run's measurement uncertainty: the feed-model spread plus the late scatter.
     - τ² is the between-run variance (method of moments).
     - R_group is the precision-weighted mean.
   - No pressure term (Spearman |ρ| ≤ 0.21, p ≥ 0.33).
   - k_r is at most 1, because s_g is already at its maximum, 1 − Swi. A run with R_r > 1 is reported.
3. **Distribution by probability.**
   - In each frame, s_g(x) = (1 − Swi)·min(1, p(x)/p*_r,t), where p is the stored pixel-head gas probability.
   - The one scalar p*_r,t is solved per frame so that the frame's free-gas volume equals k_r times its B1 value. More certain gas keeps more saturation.
   - The pore water freed this way holds CO2 at local solubility, as the Swi water does. The CO2 moves from gas to dissolved, and the total changes by about −2 % of the gas-pixel mass.
4. **What is reported, without refitting a or β:**
   - checks 1–3 with F on;
   - overflow against model gas over time, per run;
   - gas and dissolved mass, per run and time.

## 3. Rules

- **Lock gates first.** Nothing here touches a pinned module, the R3 inputs or receipts, or the plan of a running gate. Heavy runs wait while the same-host39 replay runs.
- **Laptop.** The code must also run on an ordinary laptop without a dedicated GPU.
- **Files.** Everything is written to new folders. Round 4, P1 final 3 and all existing results stay untouched. Nothing is deleted.
- **SMB errors.** A single SMB file or claim error waits and retries 2–3 times (N111).
- **Paths.** Use X: or the UNC path, not Z:.
- **Sharing.** Nothing is shared outside the project yet.

## 4. Time estimate

The estimate assumes the lock this afternoon and the decider's approvals during the day.

| Step | Estimate |
| --- | --- |
| A1 | Today, after the decider approves the map |
| B, 80-image job | One night with both machines (about 3,090 frames) |
| B, today's 1,603 frames | About half a night |
| C | Starts on the fit runs while B continues. Fit extraction about 4 h, LORO about 2 h. |
| D | About 2 h for the held-out frames, then the scores follow |
| E | A night |

FF_AC can then be ready for analysis around Thursday morning.
