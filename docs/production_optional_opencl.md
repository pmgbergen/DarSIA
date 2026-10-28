# Optional N1 OpenCL execution

The CPU queue backend can accelerate `selected_n1_fields` internally. Native
decoding, geometry, colour paths, phase inference and float64 integration keep
their existing implementations. The separate mass-materialization OpenCL
kernel has different scientific inputs and is not reused for native N1.

The SHA-bound N1 run configuration accepts `opencl_acceleration`:

```json
{"format":"darsia-opencl-n1-v1","mode":"auto","device_id":null}
```

`mode: "cpu"` explicitly selects the original CPU operation. No device
environment variable overrides this configuration. The absent block means
CPU execution with no discovery or additional output facts. Round-five
production configurations declare `auto` explicitly. Direct standalone
`PixelMath` calls keep CPU execution until explicitly configured. The configuration descriptor already binds these
settings by SHA; the released proof uses its explicitly bound automatic control.
The staged planner copies this block into the phase-one statistics template;
the phase-two sealer retains it from the actual bound phase-one configuration.

On the first call, the original CPU result is returned. Each compatible GPU
replays the entire operation twice using at most 262,144 pixels per tile. Every
output byte must match the CPU, including dtype, NaN payloads and signed zero.
Only a device whose slower complete operation beats the observed CPU call is retained.
Both probing and later calls time the same finite guard, operand conversions,
six complete output allocations, tile transfers, execution and output scatter.
Byte comparison overhead and discovery/context/compilation/device allocation
are recorded separately. This selection
measurement is not an end-to-end throughput claim. Missing runtimes, missing
devices, unsupported precision, numerical differences and runtime failures
fall back to the original CPU operation without a prompt.

The original NumPy and existing CUDA N1 formulas define the kernel operation
order. OpenCL contraction is disabled, optimization is disabled and explicit
intermediates preserve the selected working precision. No geometry or reduction
is performed on the device. A tile uses ten floating buffers plus two byte-mask
buffers: at most 11,010,048 bytes in float32 or 21,495,808 bytes in float64,
with no full-frame device allocation. Actual application device
buffer bytes are recorded separately from process memory; they are not peak
RSS or total driver usage. Probe candidates never hold device buffers concurrently;
the selected session is rebuilt after all candidates close. Peak successful
application buffer allocation is recorded separately; a failed allocation marks
its actual peak unavailable. Host memory includes the original six CPU arrays
and six complete probe outputs, tile staging and up to three complete converted
operand arrays. Those explicit array byte amounts are recorded separately from
the actual process peak; they are not a complete host-memory accounting.

Eligibility requires the working precision's round-to-nearest, denormal and
INF/NaN capabilities. Every accelerated call requires finite converted
`prepared`, `swi`, solubility, gas density, gain and factor. A nonfinite call
uses the unchanged CPU operation and is counted. The first call's byte-verified
replays, later guarded calls and proof-verified accelerated calls have separate
counters. A lock serializes a provider's computation and close. PyOpenCL's own
compiler cache is disabled with `cache_dir=False`; driver JIT caches may still
grow; the common proof must inventory and retain those files before release.

`pixel_backend_facts.opencl_n1` records the control, compatible candidates,
platform/device/driver, byte comparisons, first-call times, actual accelerated
call count/time, last execution backend and application buffer bytes. The
logical queue backend remains CPU. The output records do not claim production
permission. On Windows the facts snapshot also records the operating system's
current-process lifetime peak working set, with that scope stated explicitly;
elsewhere that field stays unavailable. It is distinct from device-buffer bytes.
Only an explicitly declared block adds these facts. Phase one includes them in its existing `native_statistics_runtime.json`
sidecar, anchored by the frame inventory. Phase two keeps them in the existing
N1 per-frame result metadata. All three existing numerical runners call the
provider's `close()` in their `finally` blocks.

An accepted common proof profile supplies the runtime's qualified device list
and proved CPU fallback. Device identity includes the OpenCL platform/version,
device/index, driver and global memory. A device absent from that list is
filtered before a context or buffers are allocated, retaining CPU execution.
After a driver update its identity no longer matches, so that device uses CPU
until the new identity is qualified. A worker-bound explicit `auto` control
requires the accepted device policy; a missing policy is allowed only by the
explicitly released common-proof binding. Unbound local declared calls can probe.
The independent common proof has no accepted-profile device list and probes
candidates for its actual comparison; that fact alone grants no production use.

The released common proof additionally enables the runtime's proof-only CPU
comparison flag. Every accelerated P1/statistics and P2 call then evaluates the
original CPU operation and compares all six arrays byte for byte. A mismatch
fails the proof; it cannot become a successful CPU fallback. CPU operation,
device operation and verification times are recorded separately. The proof's
actual measured peak includes temporary CPU reference duplicates, whose array
bytes are also recorded. Production has this flag off. An ordinary-stage
speed estimate derived from operation times is not measured end-to-end
production throughput, and the shared proof wall time includes verification.

The released proof runs one automatic P1, calibration and P2 chain with
unchanged scientific input bindings and the proof comparison flag enabled.
Every accelerated N1 call supplies its original CPU reference, byte comparison
and CPU/device timings. The accepted comparator binds those observations and
all three stages' scientific outputs. The bound report records the chain's
actual native wall time, host and device memory scopes, and usable devices per
generic host label. N1 operation speedup retains its measured scope; the chain's
wall time includes verification. No production run or qualification has been
started by this patch.
