# Method core

This document describes the code in this Git commit. Scientific qualification is recorded separately in the final validation gate report, which binds the exact clean commit, environment and per-image evidence. The method description itself is not a gate result.

The same library runs a fixed series from configuration on each worker. The code contains the image operations; the series DATA contains selection, source and geometry bindings, frozen calibration, model heads and rule parameters.

## Processing order

1. `pipeline/`: admit the original selected acquisitions and preserve their source-bound chronological order. Selection and every input descriptor are series DATA.
2. `native/`: decode the original camera RAW, register it to the admitted camera geometry, and apply the frozen light and colour corrections. Scientific log RGB and source validity remain separate from display images and photographic support.
3. `phase/`: evaluate the frozen optical and regional heads, then apply the ordered geometric rules. The rules receive their physical distances, areas and thresholds explicitly. The photographic support mask does not come from predicted labels.
4. `geometry/`: supply the admitted rig canvas, hardware exclusions, homographies, pixel areas and physical correction. The cleanup preserves the existing domain conventions; rebuilding the domain is a separate, subsequent method step.
5. `mass/`: apply the frozen N1 calibration to the colour paths, combine concentration with measured geometry and transport weights, and retain the original mass, estimate and unknown-capacity categories. The measured S_wi values are configuration data. Undefined statistical observations remain undefined.
6. `pipeline/`: process the ordered selection through these stages, publish one completed frame at a time, then write scalar tables and a whole-run completion.

## Chronological history

History belongs to camera acquisitions, identified by their original source digests and ordered timestamps. A resumed run restores the last completed acquisition's source arrays and homography. It does not use its predicted labels as the next classifier's optical input. The reference acquisition is distinct from the first selected frame.

## Shared worker protocol

One queue task owns one whole run. Inputs are read directly from UNC storage. Claiming uses an atomic rename into the machine's claim directory. A worker writes only its attempt's result files; terminal files are immutable. The controller aggregates genuine completion records.

At startup the worker verifies the locked Git commit, clean tree and environment lock. `--watchdog` keeps a separate parent process that restarts an absent child after the original progress lease expires. Completed frames are verified and reused from their original producers; incomplete frames and earlier attempts remain preserved.

```text
python -m darsia.fluidflower.pipeline.runner --series <UNC-series.toml> --series-sha256 <SHA256> --job <UNC-job>
python -m darsia.fluidflower.pipeline.worker --job <UNC-job> --backend cpu --workers <N> --watchdog --exit-when-idle
python -m darsia.fluidflower.pipeline.aggregate --job <UNC-job> --output <fresh-UNC-aggregate.json>
```

CPU and CUDA use the same declared production precision. Metadata records the actual precision and producer. Float64 checks are explicit separate jobs, with the same configuration interface.

## Retained SDK and tools interfaces

The tools package includes protected callable SDK adapters and saved-review/training readers alongside CLI entrypoints. The cleanup removes classified historical launch and calculation branches while preserving those approved interfaces.

`src/darsia/fluidflower/tools/phase_rig_registration.py` remains the callable optional sand-registration adapter on DarSIA's existing shape-corrected grid. It uses explicitly enabled sidecar profiles and preserves its existing missing-sidecar behavior. Its `DEFAULT_POLICY` series-config locator remains in this compatibility SDK adapter, outside the parameterized method core. Class-1 retention includes its callable behavior; the tools package is not classified wholesale as CLI-only.

## Byte-locked CUDA provider and external method documentation

`src/darsia/fluidflower/mass/mass_cuda.py` remains byte-identical to the locked provider: SHA256 `ebc695edf36bc308b7f994d589c0de74660c1f692c8d09b25d764ba351a9e5ea`, 26,135 bytes. Its historical in-file wording and runtime fact fields remain unchanged. The package initializer supplies runtime __doc__ help strings for exactly the 23 methods whose original in-file docstring is absent; the function objects, numeric codeobjects and provider bytes stay unchanged. No CuPy is imported or device allocated by this CPU-side documentation attachment. The table describes the same physical-purpose contracts:

`\\<nas>\<share>\FF_AC\DarSIA\code\validation\cleanup_20261014\step6_readonly_preparation\qualified_cuda_public_API_documentation_exception_v1.json` — SHA256 `a95dcb6307bc3ea8a8accdcc00b7e35081c43cb426b08a2c9a53b0e985a4166c`, 4,006 bytes.

The provider has a host ABI with explicit uploads, scoped device kernels, synchronization and returned host arrays. Working dtype, continuous sampling, frozen operands and float64 host reductions retain the locked conventions. Its memory-admission/scope guards and input errors remain unchanged. Final source/gate adoption and GPU permission require actual qualification receipts. This documentation preparation performs no GPU work and makes no throughput or new qualification claim.

| Byte-locked public method | Actual signature | External method description |
|---|---|---|
| `CudaPixelKernels.__init__` | `self, precision, *, device_id=0, pool_limit_bytes` | Select declared precision, explicit device and measured pool budget; validate memory and initialise TAB32 with FMA contraction disabled. |
| `CudaPixelKernels.array` | `self, value` | Admit/convert device arrays only inside an active allocation scope. |
| `CudaPixelKernels.upload` | `self, value` | Use the scoped device-array admission path for host inputs. |
| `CudaPixelKernels.download` | `self, value` | Return host arrays with a blocking device-to-host copy. |
| `CudaPixelKernels.facts` | `self` | Report actual device/runtime/precision/pool/events; preserve historical eligibility and permission fields. |
| `CudaPixelKernels.close` | `self` | Synchronise the owned device and release unused blocks from its pool. |
| `CudaHostPixelMath.__init__` | `self, precision, *, device_id=0, pool_limit_bytes` | Construct the device backend and corresponding NumPy working dtype. |
| `CudaHostPixelMath.pixels` | `self, value` | Convert host pixels with np.asarray to the declared dtype. |
| `CudaHostPixelMath.require_pixels` | `self, value` | Require an existing NumPy array in the exact working dtype. |
| `CudaHostPixelMath.decode_log_rgb` | `self, storage` | Require float32 frozen log storage and decode via a scoped upload/compute/download stage. |
| `CudaHostPixelMath.warp_continuous` | `self, values, matrix, shape_hw, *, cv2_module` | Cast continuous input before arithmetic, admit positive native height/width, and invoke TAB32 sampling. Categorical transport remains separate. |
| `CudaHostPixelMath.reference_normalized` | `self, current, baseline, clear, support` | Admit working current/baseline/clear arrays and boolean support; return two host normalization results. |
| `CudaHostPixelMath.relative_rgb` | `self, current, baseline, *, option='plain'` | Admit working arrays and apply the explicit relative-RGB option through the device boundary. |
| `CudaHostPixelMath.per_facies` | `self, signal, labels, models, *, ignore_labels=()` | Validate the integer facies grid, evaluate supplied models on host subsets and scatter exact-dtype outputs. |
| `CudaHostPixelMath.fit_color_path` | `self, colors, supports, path_distances` | Project working RGB onto supplied frozen colour supports/distances through the device boundary. |
| `CudaHostPixelMath.interpret_color_path` | `self, parametrization, path_distances, values` | Interpolate supplied colour values at the working parametrization through the device boundary. |
| `CudaHostPixelMath.piecewise_linear` | `self, signal, supports, values, *, endpoint='error'` | Apply supplied supports/values with the explicit error or clamp endpoint rule. |
| `CudaHostPixelMath.titration` | `self, signal, flash` | Evaluate the supplied flash/titration binding through the device boundary. |
| `CudaHostPixelMath.hydrostatic_fields` | `self, **kwargs` | Forward explicit physical keyword parameters to the scoped hydrostatic calculation. |
| `CudaHostPixelMath.selected_n1_fields` | `self, prepared, **kwargs` | Require working prepared pixels and forward explicit selected-N1 physical parameters. |
| `CudaHostPixelMath.integrate` | `value, weight, mask, *, space_dim=2` | Validate the boolean domain; form weighted host float64 values and reduce axes in the original order with float64 sums. |
| `CudaHostPixelMath.facts` | `self` | Add host-ABI, CPU facies scatter/reduction and per-stage transfer facts to device facts. |
| `CudaHostPixelMath.close` | `self` | Close the owned device backend. |

The existing documented `scope` and device `warp_continuous` methods retain their in-file documentation. The table covers exactly the 23 originally undocumented public methods whose runtime help strings are attached by the initializer. Their original raw AST None entries are retained explicitly, not falsely counted as in-file literals.

## Saved Studio views

Studio opens saved candidates, labels, annotations, layers and mass results. Its metadata readers admit original source/model bindings without executing historical numerical scripts. Archived validation sources are provenance and are not production implementations.

## Verification and accounting

The numerical gates compare arrays and scalars exactly, with equal NaN locations, and PNGs per pixel. Only explicitly listed origin fields are excluded. Gate reports retain the actual executable commit, environment, complete per-image results and process retirement evidence. `INVENTORY.md`, `REMOVED.md` and the validation index record source/result archive bindings, constant origins and classified Studio actions. The package counts in INVENTORY use the immutable Python blobs in this commit; the actual Step 6 verdict is recorded by the external gate report after the required receipts exist.

## Configuration and code placement

The tracked environment lock is `requirements-worker.lock`. Run from a clean checkout of the job commit in that environment, with this checkout installed or its `src` directory on `PYTHONPATH`. Both machines use the same module command; no runtime tree, generated start program or per-run code installation is needed.

The NAS `code/pipeline` tree contains only configuration, models, masks and locks. Validation scripts and the exact historical producer sources are under `code/validation`. Original result files and partial attempts remain at their recorded paths. Historical code is never imported to display a saved result.

`INVENTORY.md` records every archived source version and constant origin, including the actual 72 result-bound archives and 52 retired NAS control byte versions. Final package counts use tracked Python blobs from the actual baseline and final commits; the source-removal table and three-commit package counts are included in INVENTORY. `REMOVED.md` distinguishes deleted implementations from retained annotation and review functionality. The gate report supplies executable commit and actual per-image evidence; source compilation or a live controller alone is not a gate result.


## Public API unit evidence

The integrated candidate passed 538 unit tests, including 84 new cases (82 public-API contracts and two runtime-documentation contracts). All 398 public method-core APIs were observed in passed cases by a source-bound Python call profile. The CUDA cases used actual available CuPy for 14 small float32/float64 GPU tests; missing-device skips would not count as passed API calls. This is unit evidence, not a new scientific equivalence qualification or throughput benchmark. The final clean-commit gate report separately records numerical and Studio qualification.

The provider file retains its exact original bytes and 23 raw AST docstring slots remain None. The mass package initializer attaches physical-purpose and parameter runtime __doc__ text to those same function objects without altering co_code/co_consts, signatures or the two existing in-file docs. All 25 CUDA public methods therefore have runtime help strings; a fresh CPU import loads no CuPy or device allocation. This is runtime documentation, not a claim that literal strings were inserted into the byte-locked file.
