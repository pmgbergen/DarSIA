# Round 5, variant F: runs without usable overflow, and how F enters C, D and E

N254 tasks 2 and 3. Status 21 Oct 2026, about 01:30 Oslo. Read-only: nothing was changed.
Source: the overflow review `ffac_overflow_review_20261018\results\overflow_review.json`, SHA-256
f39dd5568d110c4dbbf45855d93573198838b7b47171b3b42dc4251bdebba8ee (the round-5 spec, F.1), and its QA result
`results\hidden_gas_tests.json` (SHA-256 2376115c…, the QA rule in `scripts\hidden_gas_tests.py`, lines 23-35).

## Sammendrag

- 24 av de 39 analyserte runsene har brukbar overflow. De 15 andre får gruppeverdien, slik spesifikasjonen sier.
  - 7 runs faller på QA-reglene (forsyningen stopper for tidlig, støy, drift, eller at forsyningslinja ikke treffer
    målingene før injeksjon).
  - 8 runs har ingen brukbare vektdata.
  - 2 av de 12 runsene som holdes utenfor tilpasningen (AC32, AC47) er blant de 15.
- Anbefaling for F:
  - a og β tilpasses med F slått på (steg C), slik at kalibreringen bruker samme gassledd som produksjonen. Det er samme
    prinsipp som for trykket (N246).
  - B1 uten F tilpasses også og rapporteres ved siden av, så effekten av F synes per run og tid.
  - Kontrollene D og produktene E bruker F.
  - Gruppeverdien regnes bare fra de 14 brukbare tilpasningsrunsene og brukes uendret i C, D og E.
- Fem ja/nei-spørsmål står nederst.

## 1. Runs without usable overflow (spec F.1)

QA rule (written before the pooled numbers): the supply runs at least 10 h; the late residual SD is at most 1.5 g;
|drift| is at most 3 g/h²; the late feed line meets the clean pre-injection readings within 50 mL; and the
pre-injection readings are consistent (|slope| < 600 g/h).

| Run | Role in round 5 | Reason | Evidence (overflow_review.json f39dd556 / QA result 2376115c) |
|---|---|---|---|
| AC17 | fit | No scale review | `skipped`: "no scale review" |
| AC18 | fit | No scale review | `skipped`: "no scale review" |
| AC23 | fit | No scale review | `skipped`: "no scale review" |
| AC26 | fit | Only 2 reviewed readings | `skipped`: "only 2 reviewed readings" |
| AC29 | fit | No scale review | `skipped`: "no scale review" |
| AC31 | fit | Drift and a feed line that misses the start | drift −3.212 g/h²; the late line misses the pre-injection readings by −139.9 mL; late feed 273 g/h against a median of 387 g/h |
| AC32 | held-out | Weight data flagged invalid | `skipped`: "weight data flagged invalid (No overflow?)" |
| AC38 | descriptive | No scale review | `skipped`: "no scale review" |
| AC40 | fit | No reading before injection | `skipped`: "no pre-injection reading (feed B impossible)" |
| AC45 | fit | Inconsistent readings before injection | pre-injection slope −923.29 g/h |
| AC46 | fit | Large drift; the feed line misses the start | drift −24.279 g/h²; misses by −2053.67 mL; late feed 139 g/h; one pre-injection reading |
| AC47 | held-out | The feed line misses the start | misses by −111.95 mL |
| AC48 | fit | Late noise; the feed line misses the start | late SD 2.19 g; misses by −66.07 mL |
| AC50 | fit | Supply stops early, noisy | supply stops at 4.83 h; late SD 10.14 g; drift −14.635 g/h²; misses by −56.35 mL; 39 readings; weight data "unreadable" |
| AC53 | fit | Supply stops early, noisy | supply stops at 5.2 h; late SD 10.63 g; drift −15.41 g/h²; misses by −78.97 mL |

- **Usable (24):** AC19, AC21, AC22, AC25, AC28, AC30, AC33, AC34, AC35, AC39, AC41, AC42, AC43, AC44, AC49, AC52, AC54,
  AC55, AC56, AC57, AC58, AC59, AC60, AC61.
  - Of these, 10 are held-out runs (AC19, AC21, AC22, AC25, AC35, AC41, AC44, AC52, AC56, AC60).
  - The other 14 are fit runs: AC28, AC30, AC33, AC34, AC39, AC42, AC43, AC49, AC54, AC55, AC57, AC58, AC59, AC61.
- **Outside the analysis:** the chemical outliers AC20, AC24, AC27 and AC51 (spec section 1).
- **Watch:** AC34 and AC41 have only one reading before injection.
  - In AC34 it lies 9 min before the start, so the ~20 mL pH-sampling drop falls after the anchor, and R comes out
    about 10–15 % low unless the anchor is moved after the drop (review addendum).
  - AC41 has the same single reading; its anchor needs the same check.
- **Already in the source:** AC33 and AC35 each had one misread reading. Both are corrected (1584.7 g and 366.5 g) and
  included in f39dd556.

## 2. How F enters C, D and E (proposal)

**Facts the proposal rests on:**
- The model's free gas (Σ s_g·PV over gas pixels, with s_g = 1 − Swi) depends on the phase maps (B), the map and the
  geometry (A2). It does not depend on a or β.
- So k_r and p*_{r,t} (spec F.2-F.3) can be computed after B, before the fit. There is no circularity.
- With F, gas falls by about 12 % (R at 2.25 h is 0.88). The freed pore water holds CO2 at local solubility, so the
  total changes by only about −2 % of the gas-pixel mass, but the split moves toward dissolved CO2.
- The gain a is fitted so that the total matches the dose. If F is applied after the fit, the total no longer matches
  the dose by construction, and a and β belong to another gas term than the products.

**Proposed order:**

| Step | With F? | What |
|---|---|---|
| After B | — | Compute R_r, σ_r, w_r and k_r per run and p*_{r,t} per frame from the stored gas probability. R_group comes from the 14 usable fit runs only. Report any run with R_r > 1. |
| C | Both | Fit B1F (F on: the production calibration) and B1 (F off: the reported reference). The feature extraction is shared, so the second fit adds about the LORO time (~2 h). B0 stays the baseline. Freeze B1F and B1 separately (files and SHA-256). |
| D | F on | Score checks 1–3 once on B1F, the production configuration. Report B1 and B0 beside it, unscored for the decision. If B1F fails and B1 passes, you decide (F may be the cause). If both fail and B0 passes, you decide, as the spec says. |
| E | F on | Products with B1F. Store k_r, R_r, w_r, σ_r and p*_{r,t} with the products and receipts. Give B1 without F as mass tables per run and time (gas, dissolved, total), not as a second set of per-pixel maps. |

**Why the group value comes from the fit runs only:** in C it sets the gas of the 12 fit runs without usable
overflow. Taking it from held-out runs would let held-out data into the fitted a and β. Using the same value in D and
E means that what is checked is what is produced. The 10 held-out runs with usable overflow still use their own
overflow, as production does, because F is a measurement rule, not a fitted parameter.

**Unchanged:** the phase maps (F changes only s_g); the round-5 checks and stops; k_r ≤ 1; no pressure term in k_r
(Spearman |ρ| ≤ 0.21).

## 3. Ja/nei-spørsmål

- **F1:** Skal a og β tilpasses med F slått på (B1F), og B1 uten F tilpasses i tillegg som referanse? Anbefalt: ja.
- **F2:** Skal kontrollene D score B1F én gang, med B1 og B0 rapportert ved siden av, og skal du avgjøre hvis B1F feiler
  mens B1 består? Anbefalt: ja.
- **F3:** Skal produktene E lages med F, med F-faktorene per run og bilde lagret, og B1 uten F levert som massetabeller
  (ikke egne pikselkart)? Anbefalt: ja.
- **F4:** Skal gruppeverdien regnes bare fra de 14 brukbare tilpasningsrunsene og brukes uendret i C, D og E? Anbefalt: ja.
- **F5:** Skal AC34 og AC41 få gruppeverdien hvis ankeret ikke kan legges etter pH-prøvefallet? Anbefalt: ja.
