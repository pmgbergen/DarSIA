# Results summary for the talk: regularized heterogeneous Beckmann problem

This is a brief for building slides. It lists which figure goes with which message, the numbers to quote, and what the figures must not be used to claim.

All paths are relative to `talk/` in the DarSIA repo (`/home/trygve/PhD/Kode/DarSIA/talk`).

- **Figures:** `results/figures/*.pdf` are vector files for the deck. Each has a `.png` preview next to it.
- **Raw data:** `results/*.csv`.
- **Rates computed from the data:** `results/figures/observed_rates.txt`.
- **Regenerating:** run `python plot_results.py --png` from `talk/`. The wall transport-density figure comes from `python plot_wall.py`.

## 1. Setup (for the method and setup slides)

**The problem.** Compute the Wasserstein-1 distance with a heterogeneous cost (mobility κ) by solving the Beckmann problem:

  min_q ∫ κ⁻¹ |q| dx  subject to  div q = μ − γ.

The non-smooth |q| is replaced by Ψ_ε(s) = s − ε ln(1 + s/ε), with Ψ_ε′(s) = s/(s+ε).

**Discretization.** RT0 face fluxes and P0 cell potentials on a 2D Cartesian grid, with h = mesh width and n = cells per side. Three variants differ in where |q| is evaluated:

| Variant | Name in figures | How \|q\| is evaluated |
|---|---|---|
| N | "(N) normal only" | Absolute value of the normal flux on each face. |
| F | "(F) face reconstr." | Normal flux on the face combined with the average of the 4 adjacent tangential fluxes. |
| C | "(C) cell reconstr." | Cell vector built from the averaged x- and y-face fluxes. |
| reference | "DarSIA (RT, cell)" | DarSIA's existing unregularized Newton solver. |

**Nonlinear solvers:**
- **L-scheme:** projected gradient with step 1/L, where L = 1/(κ_min ε). Each step is one fast DCT Poisson solve.
- **Lagged diffusivity** (also known as Kačanov or IRLS): each step solves a weighted Darcy problem with mobility κ(|q_prev| + ε). The linear solver is MINRES, preconditioned with AMG on the Schur complement.
- **Options for lagged diffusivity:**
  - Anderson acceleration of depth 5 ("AA"). A safeguard keeps the AA step only if it lowers the energy at least as much as the plain step.
  - ε-continuation ("contin."): halve ε repeatedly, warm-starting each stage from the previous one.

**Test cases** (exact values are known for all three):
- **box_axis:** a uniform box of side 0.25 on [0,1]² is translated by (0.25, 0). W = 0.25.
- **box_diag:** the same box translated by (0.25, 0.25). W = 0.3536.
- **wall:** domain [0,8]². Two 2×2 boxes centred at (2,4) and (6,4). A wall at x=4 of height 6 and thickness 0.4 makes crossing cost K=8, so all mass goes around the wall. W = 26.568, from a semi-analytic formula.

All runs use tolerance 1e-6 on the relative KKT residual.

## 2. Recommended slide sequence

### Slide A: what the transport looks like (motivation)
**Figure:** `results/wall_test_flux_F.png`

This is the transport density |q| for the F variant. Columns are wall strength K = 0, 1, 3, 8; rows are ε = 0.05 (top) and ε = 5e-4 (bottom). n = 40.

**Message:** as the wall gets stronger, the mass changes from crossing straight through, to splitting, to going entirely around the wall ends. Larger ε smears the transport density.

**Caveats:**
- This PNG has titles, unlike the other figures, and no PDF version exists. Regenerate `plot_wall.py` without titles if the deck style requires it.
- This figure uses an older wall geometry from `wall_test.py`: thickness 2 cells and κ⁻¹ = 0.5·K/h. Show it as an illustration only, not as the wall test case with exact value 26.57.

### Slide B: the normal-only discretization converges to the wrong distance
**Figures:** `results/figures/exp1_eps_convergence.pdf` (dashed N line) and `results/figures/exp1_wall_h_convergence.pdf` (N line)

**Message:** with |q| measured only through normal fluxes (N), the distance converges to the ℓ¹ (Manhattan) distance, not the Euclidean one. Reconstructing the flux vector (F or C) is necessary.

**Numbers:**
- box_diag: N gives 0.5000, which is exactly the ℓ¹ distance, against the true 0.3536. That is a 41 % error, and it does not shrink as ε → 0 or h → 0.
- wall: N gives about 36.2 against the true 26.57, a 36 % error. It is flat under refinement.
- N is correct only when transport is axis-aligned: box_axis, and the wall with K=0.

### Slide C: the regularization error is O(ε)
**Figure:** `results/figures/exp1_eps_convergence.pdf` (box_diag, n=256, continuation from ε=1 down to 1e-5)

**Message:**
- The ε-error decays linearly in ε, with relative error ≈ 1.0–1.1·ε for both F and C.
- Below ε ≈ 1e-4, F levels off at its h-error, about 3e-5 at n=256.

**Numbers:** F at n=256: ε = 9.8e-4 → 1.07e-3; 4.9e-4 → 5.5e-4; 2.4e-4 → 2.9e-4; 1.2e-4 → 1.55e-4.

**Caveat:** the C curve stops at ε ≈ 1e-3. That run hit its 30-minute time budget. It is not a failure of the method.

### Slide D: how the error splits between ε and h (strongest accuracy slide)
**Figures:** `results/figures/balance_h_eps_wall.pdf` and `results/figures/balance_h_eps_box_diag.pdf`

Each figure plots error against ε for three mesh sizes; the left panel is F and the right panel is C. The plateau at small ε is the discretization (h) error, and the sloped part is the ε-error. Dotted vertical lines mark eps*, the ε at which the two errors are equal.

**Message:** error ≈ c₁·ε + c₂·h^r. Choose ε proportional to h (or h²). Pushing ε far below that buys no accuracy.

**Numbers:** "h-error" below is the error extrapolated to ε → 0.

| Case | Variant | h-error, coarse → fine | Observed order |
|---|---|---|---|
| wall (n = 40, 80, 160) | F | 9.0e-2 → 4.4e-2 → 2.2e-2 | 1.0 |
| wall (n = 40, 80, 160) | C | 2.3e-2 → 1.15e-2 → 5.7e-3 | 1.0; about **4× smaller error than F** |
| wall (n = 40, 80, 160) | DarSIA | 3.7e-2 → 1.8e-2 → 9.1e-3 | 1.0; between F and C |
| box_diag (n = 32, 64, 128) | F | 1.85e-3 → 4.3e-4 → 1.0e-4 | **2.1** |
| box_diag (n = 32, 64, 128) | C | about 2e-7 (none measurable) | none; the right panel shows a single line |
| box_diag (n = 32, 64, 128) | DarSIA | 4.9e-3 → 1.5e-3 → 4.5e-4 | about 1.7 |

- **eps*** on the wall: about 0.3·h for F and about 0.07·h for C. On box_diag (F), eps* scales like h².
- **First order on the wall** is expected: the wall corners are singular and the cost κ⁻¹ jumps there.
- **Why C is exact on box_diag** is a hypothesis, not verified: averaging to cells is exact for the piecewise-constant translation flux.
- **Data sources for the table:** the F and C rows come from `balance.csv` and `observed_rates.txt`. The DarSIA rows come from `h.csv`. DarSIA's finest wall point (n=160) stopped at its iteration cap; it is plotted as a hollow marker.

### Slide E: iterations against ε
**Figures:** `results/figures/exp2a_iterations_vs_eps_wall.pdf` (wall, n=40) and `results/figures/exp2a_iterations_vs_eps_box_diag.pdf` (box_diag, n=64)

**Message:**
- The L-scheme needs iterations growing like ε^-1.4 to ε^-1.8. It hits the 200 000-iteration cap (hollow squares) at ε = 0.003 on box_diag, and already at ε ≤ 0.01–0.03 on the wall.
- Lagged diffusivity needs 10²–10³ iterations.
- Anderson acceleration makes it nearly insensitive to ε. On the wall, F needs 16 → 47 iterations as ε goes from 0.3 to 0.003.

**Numbers:**
- Measured L-scheme slopes: box_diag N/F/C = −1.79 / −1.47 / −1.67; wall N/F = −1.78 / −1.38.
- Plain lagged versus lagged + AA at ε = 0.003: box F 289 vs 101; box C 496 vs 166; wall F 175 vs 47; wall C 352 vs 67.

**Caveats:**
- The dotted ε^-2 line is only a reference. The measured slopes are shallower (above).
- The wall C slope (−0.86) is distorted because most of its points are capped; don't quote it.
- **Don't present continuation as an improvement.** In this setup "lagged + AA + contin." is always slower than plain "lagged + AA". It takes 3–7× more iterations and time for the same answer (`cont_cmp.csv`), because every intermediate ε stage is solved to the full 1e-6 tolerance.

### Slide F: iterations against mesh size
**Figure:** `results/figures/exp2_iterations_vs_h.pdf` (box_diag, ε=0.03, n = 32 to 256)

**Message:** both nonlinear solvers need about the same number of iterations on every mesh.
- L-scheme: about 8.4k (N) and about 2.3k (F, C).
- Lagged + AA: settles at about 90 iterations.

**Numbers:** MINRES needs about 24 iterations per nonlinear step for F, also independent of the mesh. For C it needs 120–175 and grows slowly. The C preconditioner is the weak point; a hypothesis is that face-flux checkerboard modes are invisible to the cell average.

### Slide G: cost against accuracy on the wall (strongest efficiency slide)
**Figure:** `results/figures/workprec_wall.pdf` (ε = 1e-4, lagged + AA without continuation; point labels are n)

**Message:** C costs more per iteration but gives the best accuracy for a given run time. C at n=80 reaches the accuracy of F at n=320 roughly **11× faster**.

| | n | Relative error | Time |
|---|---|---|---|
| F | 40 / 80 / 160 / 320 | 9.1e-2 / 4.4e-2 / 2.2e-2 / 1.1e-2 | 2 s / 8 s / 44 s / 291 s |
| C | 40 / 80 / 160 | 2.4e-2 / 1.2e-2 / 5.9e-3 | 5 s / 26 s / 191 s |

### Slide H (optional, needs care): run time against problem size
**Figure:** `results/figures/exp2b_time_vs_N.pdf` (box_diag, N = 1.6e4 to 1.05e6 cells)

**Message that is safe to make:**
- Lagged + AA (ε=0.05) takes roughly constant iterations (about 61–65), and its time grows about linearly in N up to 10⁶ cells. That is close to optimal.
- For F, it is about 2× faster than the L-scheme at every N.

**We are faster than DarSIA at matched accuracy, but this figure does not show it.** In the figure our runs are at ε = 0.05 (about 4.5 % error) or ε = 0.005 with continuation (about 0.5 %). DarSIA solves the unregularized problem (0.05 % error at n=128). So the figure compares fast-but-inaccurate against slow-but-accurate. Make the speed claim with the matched-accuracy comparison below, taken from `h.csv`, `iter_eps.csv`, `cont_cmp.csv` and `workprec.csv`.

**Comparison at matched accuracy:**

| Case | DarSIA | Ours | Result |
|---|---|---|---|
| box_diag | n=64: 8.4 s, error 1.5e-3 | F, n=64, ε=1e-4: 2.6 s, error 5.3e-4 | 3× faster, 3× more accurate |
| box_diag | n=128: 30 s, error 4.5e-4 | F, n=64, ε=1e-4: 2.6 s, error 5.3e-4 | about the same accuracy, **about 12× faster** |
| box_diag | n=256: 145 s, error 1.3e-4 (not converged) | C, n=64, ε=1e-4: 26 s, error 1.1e-4 | **about 5.5× faster** |
| wall | n=40: 5.2 s, error 3.7e-2 | C, n=40, ε=1e-4: 5.5 s, error 2.4e-2 | same time, 1.6× more accurate |
| wall | n=80: 14.7 s, error 1.85e-2 | C: about 9 s at that error (interpolated, n=40 to 80) | about 1.6× faster |
| wall | n=160: 52 s, error 9.2e-3 (not converged) | C: about 53 s at that error (interpolated, n=80 to 160) | about equal |

**Why we win:** the speed-up comes from accuracy per mesh, not from a faster solver per cell. On box_diag, F converges like h² and C has no measurable h-error, so we reach a given accuracy on a much coarser mesh. On the wall all methods are first order, so the advantage comes only from C's smaller error constant. F is *slower* than DarSIA there: it needs n=160 and 44 s to reach DarSIA's n=80 accuracy.

**Caveats on the speed-up:**
- The codes differ: ours is standalone numpy/scipy, while DarSIA has more framework overhead. The stopping criteria also differ.
- Most of these timings come from batches run in parallel with other jobs. Only the `timing` batch ran alone.
- Phrase the claim as "≈ 3–12× faster on the translation test, on par to 1.6× faster on the wall (C)". Do not quote a single headline factor.
- A dedicated figure of error against run time that includes DarSIA would make this slide stronger. It does not exist yet.

**Other caveats:**
- The figure is visually busy.
- At n=1024 the runs stopped early: "L-scheme (F)" and "lagged + AA + contin. (N)" hit their time limits (hollow markers). The n=1024 run for "lagged + AA + contin. (F)" never ran, because the batch was stopped. DarSIA was only run up to n=256 by design.

## 3. Figure to avoid or replace
`results/figures/exp1_h_convergence.pdf` (box cases, ε = 1e-3) is dominated by the regularization error. F and C sit flat at about 1.3e-3, so the figure says nothing about h-convergence. DarSIA's axis-shift curve gets *worse* with refinement only because those runs hit DarSIA's 499-iteration cap. Use `balance_h_eps_box_diag.pdf` instead.

`results/figures/exp1_wall_h_convergence.pdf` is usable. F is cleanly first order. C's finest point (n=320, hollow marker) hit the time cap and should be ignored, or use `balance_h_eps_wall.pdf` instead.

## 4. Key take-home messages (for the conclusion slide)
1. Measuring the flux only through normal components gives the ℓ¹ distance. Reconstructing the flux (F or C) gives the correct one.
2. Error ≈ c·ε + C·h^r: r = 1 on the heterogeneous wall case. On the smooth translation, r = 2 for F and C is exact. So choose ε proportional to h.
3. Lagged diffusivity always decreases the energy (observed in every run), needs about the same iterations on every mesh, and with Anderson acceleration is almost insensitive to ε. The L-scheme is also mesh-independent, but its iterations grow like ε^-1.5.
4. The cell reconstruction (C) gives the best accuracy per second on the heterogeneous case. A better linear preconditioner for C is the natural next step.
5. At matched accuracy we are faster than DarSIA's unregularized Newton solver: about 3–12× on the translation test, and on par to 1.6× (C) on the wall. The gain comes from smaller discretization error per mesh.

## 5. Open issues (for an "outlook" slide, or to keep off the slides)
- Continuation is slower than solving directly at the target ε, because intermediate stages are solved too tightly.
- The Anderson safeguard rejects about 50 % of steps, and each rejection resets the history. The acceleration is probably not at full strength.
- The C variant's MINRES iteration counts are high and grow slowly with the mesh.
- DarSIA's reference solver often hits its 499-iteration cap, so its numbers in those runs are approximate.
