# Shared production proof

The implementation supplies a typed P1 -> C -> P2 provider and uses the existing
queue, worker, native process controller, output publisher and retirement
checks. Preparing these sources does not release a run or publish a passing
profile. There is no separate qualification for RAW rig setup or OpenCL.

## Existing entry points

1. Prepare actual phase-one DATA with
   `analysis_preparation.prepare_phase1_inputs` / `analysis_plan.write_phase1_plan`.
   The proof selection is the accepted reference-run selection. Use the final
   RAW rig models, colour paths, pressure inputs and fitting choices. Existing
   planning and `staged_jobs.create_analysis_job` create the genuine queued P1
   task; do not hand-write a substitute task or input catalogue.
   New production DATA explicitly declares `opencl_acceleration` with mode
   `auto`. The one proof chain sets `verify_every_call`, so each accelerated N1
   call compares all six outputs with the original CPU calculation. It records
   CPU, device and verification timings in that same chain. An absent block
   retains the legacy CPU path and metadata and supplies no common proof.
2. Bind the exact existing job/task/configuration, final store map, capped cache
   namespace, source/runtime closure, whole-stage output allowance and independent
   stage release into `common_qualification.CommonQualification`. The existing
   pre-import request dispatch recognizes its request format and launches the
   original public worker through `supervise_first_worker`.
3. After original queue publication, `WorkerRunIO.finalize_common` validates
   scientific joins in the original admitted worker, before capacity closure
   and its native exit inventory. It writes an immutable worker closure, whose
   pin is recorded by the actual owner's exit inventory. After native retirement,
   `close_stage` reads metadata and hashes bytes only. It rehashes every original
   admitted input with its original file identity, verifies the owner/native
   closure and combines terminal file lengths into the immutable stage close.
   Scientific libraries and anchor calculations remain inside the admitted worker.
4. Build the actual global refit using `refit_jobs.build_plans(loro=False)`,
   `write_refit_inputs` and `prepare_refit_job`. It must use the final accepted
   fitting roles and their actual statistics. A held-out application is kept out
   of the fitting rows and group value. Release the C stage through the same
   original controller only after the P1 close exists.
   Before writing the common global plan, use
   `common_calibration.bind_common_phase1` with that actual P1 configuration,
   completion, code commit and the original `reader`. It carries the exact P1
   aliases and adds closure descriptors without changing the
   fitting roles, statistics, pool, rules or parameters. The original catalogue
   declares all named predecessor files; the worker runs the original verifier.
5. Seal P2 with `staged_data.seal_phase2_configuration`, supplying the actual
   P1 completion/configuration, `frozen_refit={"completion": completion, "plan": plan}`, its frozen
   calibration and own-run anchor, plus
   `storage_profile={"profile": "lean", "colour_intermediates": "keep"}`. The task itself
   must explicitly choose `storage_profile="lean"` and format version 2.
6. Run the released P2 through the same original queue/controller. Its exact
   fallback/store bound must fit the predeclared output allowance. The close
   measures each frame, both operation receipts and retained serialization
   partials against the admitted bound. It measures once-per-run files separately.
7. Run one auto P1 -> C -> P2 chain. Its accelerated N1 calls compare all six
   selected fields with the original CPU operation on every call and stop on a
   mismatch. Keep the actual scientific outputs and accepted comparator report.
   Report CPU/device operation timings and gain alongside the actual native
   launch-to-retirement wall time. State initialization and CPU-comparison
   overhead separately. Retain process working-set observations, kernel commit
   peaks and application GPU buffer bytes with their own scopes. Missing OpenCL
   uses the original CPU fallback; no second complete proof chain is required.
8. After review and the single final full suite, an independent final release
   may call `publish_profiles`. Both profiles resolve through the complete
   original P1/C/P2 DAG. The final release checks the source validator's own
   SHA-bound validation/release record and source closure, nonempty passing
   focused/full-suite receipt contents, and
   `accepted_comparator` / `accepted_comparison`. The in-run acceptance receipt
   binds the one immutable provider, every stage's complete indexed output
   inventory and scientific closure, all six selected fields for every P1/P2
   frame, and a SHA-bound measurement report containing the actual observations.
   The report must be present and intact; this workflow adds no acceptance
   threshold for throughput or memory. No incomplete stage authorizes ordinary use.
   The existing record schema accepts that validation owner's own id, text and
   SHA; an external reviewer or answer is not required.

Ordinary jobs receive an explicit per-run device policy from that resolved
profile. Only device/driver identities that passed the actual candidate checks
can allocate OpenCL buffers. Missing, changed or unknown devices use the proved
CPU fallback. The SHA-bound run control is never rewritten or supplied through
an environment override. References and existing jobs keep the full layout.

The existing first-qualification environment redirects `USERPROFILE`, `APPDATA`,
`LOCALAPPDATA` and `HOME` under the owned service cache. The original whole-root
scanner counts nested driver JIT files as used bytes. Common proof measurements
retain complete file identities, lengths and SHA-256 before launch and after
retirement, including preexisting driver files and new compiler-cache files.
Their scan times are recorded; hashing the whole retained root can be expensive.
The live accounting default still reads only file metadata. These observations
cover the admitted cache root and make no claim about exhaustive OS writes
outside it. They remain distinct from source/runtime byte rechecks. No cache
file is removed by this workflow.

## Final inputs still required

- The final reviewed source commit and dependency/runtime closure, with the
  corrected RAW rig model payload certificates and registered geometry inputs.
- The final reference-run plan and scientific configuration, and the accepted
  statistics, fitting roles, variant settings and baseline for the global C fit.
- An independently released existing C plan and fit-statistics pool, if used for
  a finite proof before new production statistics exist. Held-out application
  rows cannot become fitting rows. No such proof pool is selected by this code.
  The current ordinary-C admission requires the exact proof-C configuration;
  it does not admit a later production refit with different inputs. Positive
  admission by a fixed fitting rule remains a separate required source change.
- The actual production store map chosen in the application, including the NAS
  results root, plus its available-space and capped-cache bindings.
- Independent finite stage releases and output allowances computed for those
  exact configurations, including worker closure and retained-cache inventory
  metadata. No actual common requests, stage completions, comparison
  report or passing production profile have been fabricated by source preparation.
- The actual accepted comparator source, completed outputs from the one proof
  chain, its in-run acceptance and measurement report, and independent final
  release. The schema refuses missing, failed or unrelated receipts; source
  preparation supplies none.

Cache authority and release metadata remain in the admitted local namespace.
The finite stage authority is the SHA-bound independent Root release spec,
including its exact request digest, scope and reviewed source closure. Product
code reads no handoff log and names no particular reference run or frame count.
A phase-two worker verifies its actual P1/C join during original input admission,
before scientific work, and rechecks it at scientific close.

Every ordinary claim resolves the original proof jobs and fixed receipt slots.
The NAS plan must record their storage locations, availability and retention,
including the accepted comparator source and measurement/acceptance reports.
Every ordinary claim checks the comparator at its bound path and SHA, so its
location is also a production dependency. Moving a proof job, comparator or
receipt invalidates eligibility until independently rebound; no path search or
silent relocation supplies a replacement.

A failed post-retirement close remains held with `metadata-close-only` recovery.
The same released request re-enters the original controller, validates startup,
then closes metadata before claim consumption or native-worker construction.
Scientific evidence and stage completion each have four finite released names:
the original slot plus `.close2.json`, `.close3.json`, `.close4.json` candidates.
Short writes and failed flush candidates are kept and counted. Recovery uses the
next unused name or exact complete retained bytes, rejecting conflicting complete
candidates. Reusing identical bytes does not claim that an earlier failed flush
was redone. Exhaustion stays held and needs a new independent slot release;
it never deletes files, overwrites a candidate or reruns science.

The C worker's newly published shared completion is admitted only through its
verified original job/index and same-byte owned candidate, in a separate narrow
publication mapping. Original input catalogue entries and admission-time file
versions are unchanged. Foreign attempts/publications and changed bytes refuse.
Scientific NAS inputs are read through their actual rooted catalogue and checked
after retirement; the pre-import profile resolver reads named proof metadata. All
new scientific output goes to a fresh owned attempt. No raw acquisition is
rewritten and no file is deleted.

The conservative fallback archive array envelope was reviewed with the initial
lean-capacity source. It still includes all legal full fallbacks and assumes no
compression or omission gain; this repair does not replace that accepted envelope
with a new numerical schema. Run-record names are shared with the original N1
writer, and phase-one additions use the original shared writer envelope. Per-frame
measurement consumes the original checkpoint producer descriptors rather than
rebuilding receipt paths. The serializer and retained-file classifier share the
same temporary-name helper.
