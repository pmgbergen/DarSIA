"""Unqualified CPU precision kernels; preparation only, no native entrypoint.

Frozen input storage, tables, parameters, metadata/source/phase/freeze checks
remain authoritative. This module neither reads assets nor grants admission.
Its explicit working precision begins BEFORE pixel arithmetic; a float64
result must not be produced by upcasting a completed float32 calculation.
Integration into the complete guarded loader is still required before use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass(frozen=True)
class PixelMath:
    precision: str
    _opencl: dict = field(default_factory=dict, repr=False, compare=False)
    _opencl_lock: object = field(default_factory=Lock, repr=False, compare=False)

    def __post_init__(self):
        if self.precision not in ("float32", "float64"):
            raise ValueError("Explicit float32 or float64 working precision required")

    @property
    def dtype(self):
        """Return the declared NumPy dtype for working image pixels."""
        import numpy as np
        return np.float32 if self.precision == "float32" else np.float64

    def pixels(self, value):
        """Convert image values to the declared working pixel dtype."""
        import numpy as np
        return np.asarray(value, dtype=self.dtype)

    def require_pixels(self, value):
        """Require an existing NumPy pixel array with the declared working dtype."""
        import numpy as np
        if not isinstance(value, np.ndarray) or value.dtype != self.dtype:
            raise TypeError("Pixel input must already have the declared working dtype")
        return value

    def facts(self):
        """Report CPU pixel, coefficient and reduction dtypes without granting execution permission."""
        record = dict(precision=self.precision, backend="numpy_cpu",
                    parameter_dtype="float64", reduction_dtype="float64",
                    input_storage_dtype="frozen_asset_encoding",
                    scientific_eligibility=False, production_permission_granted=False)
        if "control" in self._opencl:
            selector = self._opencl.get("selector")
            record["opencl_n1"] = (selector.facts() if selector is not None else
                dict(control=dict(self._opencl["control"]), used=False))
        return record

    def configure_opencl(self, value):
        """Bind optional acceleration to the run configuration before arithmetic."""
        if value is None:
            return  # Undeclared legacy configurations retain CPU bytes and facts.
        with self._opencl_lock:
            self._configure_opencl(value)

    def _configure_opencl(self, value):
        """Validate declared controls against the actual worker binding."""
        from copy import deepcopy
        from .mass_opencl import configuration
        from ..pipeline.run_io import current_binding
        control = configuration(value)
        binding = current_binding()
        policy = deepcopy(getattr(binding, "opencl_policy", None))
        if (control["mode"] == "auto" and binding is not None and policy is None
                and getattr(binding, "opencl_proof_released", False) is not True):
            raise ValueError("Worker OpenCL auto requires an admitted policy or its released common proof")
        compare_cpu = getattr(binding, "opencl_compare_cpu", False)
        if type(compare_cpu) is not bool:
            raise ValueError("Common N1 comparison mode must be explicit")
        if policy is not None and (not isinstance(policy, dict)
                or set(policy) != {"qualified_devices", "cpu_fallback_qualified"}
                or policy["cpu_fallback_qualified"] is not True
                or not isinstance(policy["qualified_devices"], list)):
            raise ValueError("Optional N1 OpenCL requires its admitted CPU fallback and devices")
        if "selector" in self._opencl and (self._opencl["control"] != control
                or self._opencl.get("policy") != policy
                or self._opencl.get("compare_cpu") != compare_cpu):
            raise RuntimeError("N1 OpenCL control cannot change after computation")
        self._opencl["control"] = control
        self._opencl["policy"] = policy
        self._opencl["compare_cpu"] = compare_cpu

    def close(self):
        """Release only this provider's optional device buffers."""
        selector = self._opencl.get("selector")
        if selector is not None:
            selector.close()

    def decode_log_rgb(self, stored_log):
        """Keep original log storage gate; cast BEFORE exp, not after decoding."""
        import numpy as np
        if not isinstance(stored_log, np.ndarray) or stored_log.dtype != np.float32:
            raise TypeError("Frozen log-linear RGB storage must remain float32")
        working = self.pixels(stored_log)
        return np.exp(working) / self.dtype(65535)

    def warp_continuous(self, values, matrix, shape_hw, *, cv2_module):
        """Same native INTER_LINEAR/BORDER_CONSTANT warp, explicit input dtype.

        Discrete phase/facies/domain transport must retain its existing nearest
        neighbour uint8/bool path and is deliberately outside this method.
        """
        import numpy as np
        working = self.pixels(values)
        homography = np.asarray(matrix, dtype=np.float64)
        if homography.shape != (3, 3) or not np.isfinite(homography).all():
            raise ValueError("Explicit finite native homography required")
        h, w = shape_hw
        result = cv2_module.warpPerspective(working, homography, (w, h),
            flags=cv2_module.INTER_LINEAR, borderMode=cv2_module.BORDER_CONSTANT,
            borderValue=0)
        return self.require_pixels(result)

    def reference_normalized(self, current, baseline, clear, support):
        """Native measured-reference division; support is the existing guard output."""
        import numpy as np
        for value in (current, baseline, clear):
            self.require_pixels(value)
        if current.shape != baseline.shape or current.shape != clear.shape:
            raise ValueError("Registered RGB grids differ")
        if support.dtype != bool or current.shape != (*support.shape, 3):
            raise ValueError("Native boolean normalization support required")
        normalized = np.zeros(current.shape, dtype=self.dtype)
        normalized_base = np.zeros(baseline.shape, dtype=self.dtype)
        np.divide(current, clear, out=normalized, where=support[..., None])
        np.divide(baseline, clear, out=normalized_base, where=support[..., None])
        return normalized, normalized_base

    def relative_rgb(self, current, baseline, *, option="plain"):
        """Native ConcentrationAnalysis background subtraction.

        The retained common pipeline selects 'plain'. ColorEmbedding's separate
        /255 normalization is NOT part of this pipeline and is not introduced.
        """
        import numpy as np
        self.require_pixels(current)
        self.require_pixels(baseline)
        if current.shape != baseline.shape:
            raise ValueError("Registered RGB grids differ")
        if option == "plain":
            return current-baseline
        if option == "positive":
            return np.clip(current-baseline, self.dtype(0), None)
        if option == "negative":
            return np.clip(baseline-current, self.dtype(0), None)
        if option == "absolute":
            return np.abs(current-baseline)
        raise ValueError("Unknown native background subtraction option")

    def per_facies(self, signal, labels, models, *, ignore_labels=()):
        """Typed HeterogeneousModel output; original separate label masks/order."""
        import numpy as np
        self.require_pixels(signal)
        if labels.shape != signal.shape[:2] or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Exact native categorical facies grid required")
        output = np.zeros(signal.shape[:2], dtype=self.dtype)
        for label in np.unique(labels):
            if label in ignore_labels:
                continue
            mask = labels == label
            values = models[label](signal[mask])
            self.require_pixels(values)
            output[mask] = values
        return output

    @staticmethod
    def _table(values, *, name):
        import numpy as np
        result = np.asarray(values, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError(f"Nonfinite frozen {name}")
        return result

    def fit_color_path(self, colors, supports, path_distances):
        """Native segment order, projection, L1 distance and strict < tie rule.

        Frozen supports/distances remain float64; only local operands of pixel
        operations take the selected dtype. No refitting or knot modification.
        """
        import numpy as np
        self.require_pixels(colors)
        supports64 = self._table(supports, name="color supports")
        distances64 = self._table(path_distances, name="path distances")
        if (supports64.ndim != 2 or supports64.shape[1] != 3
                or distances64.shape != (len(supports64),)
                or len(supports64) < 2 or colors.shape[-1] != 3
                or np.any(np.diff(distances64) <= 0)):
            raise ValueError("Exact ordered native RGB color path required")
        supports_work = self.pixels(supports64)
        distances_work = self.pixels(distances64)
        interpretations = []
        count = len(supports64) - 1
        for segment in range(count):
            direction = supports_work[segment+1] - supports_work[segment]
            denominator = np.dot(direction, direction)
            if denominator <= 0:
                raise ValueError("Degenerate frozen color segment in selected precision")
            scalar = distances_work[segment] + (
                distances_work[segment+1] - distances_work[segment]
            ) * np.tensordot(colors-supports_work[segment], direction,
                             axes=([-1], [0])) / denominator
            # Original if-first/elif-last order matters for one segment.
            lower = None if segment == 0 else distances_work[segment]
            upper = distances_work[segment+1] if segment == 0 or segment < count-1 else None
            interpretations.append(np.clip(scalar, lower, upper))
        shape = colors.shape[:-1] + (3,)
        projected = [supports_work[s] + np.outer(
            (interpretations[s]-distances_work[s]) /
            (distances_work[s+1]-distances_work[s]),
            supports_work[s+1]-supports_work[s]).reshape(shape)
            for s in range(count)]
        closest = np.zeros(colors.shape[:-1], dtype=np.int32)
        minimum = np.full(colors.shape[:-1], np.inf, dtype=self.dtype)
        for segment in range(count):
            distance = np.linalg.norm(colors-projected[segment], ord=1, axis=-1)
            mask = distance < minimum
            closest[mask] = segment
            minimum[mask] = distance[mask]
        result = np.zeros(colors.shape[:-1], dtype=self.dtype)
        for segment in range(count):
            mask = closest == segment
            result[mask] = interpretations[segment][mask]
        result[np.isnan(result)] = 0.0  # original ColorPath.fit behavior
        return result

    def interpret_color_path(self, parametrization, path_distances, values):
        """Original ColorPathInterpolation loop; all per-pixel operands typed."""
        import numpy as np
        self.require_pixels(parametrization)
        distances = self.pixels(self._table(path_distances, name="path distances"))
        values = self.pixels(self._table(values, name="path values"))
        if len(distances) != len(values) or np.any(np.diff(distances) <= 0):
            raise ValueError("Ordered frozen color interpolation table required")
        result = np.zeros_like(parametrization)
        count = len(distances)-1
        for segment in range(count):
            if segment == 0:
                mask = parametrization <= distances[segment+1]
            elif segment == count-1:
                mask = parametrization >= distances[segment]
            else:
                mask = ((parametrization >= distances[segment]) &
                        (parametrization <= distances[segment+1]))
            weight = (parametrization[mask]-distances[segment]) / (
                distances[segment+1]-distances[segment])
            result[mask] = values[segment] + weight*(values[segment+1]-values[segment])
        return result

    def piecewise_linear(self, signal, supports, values, *, endpoint="error"):
        """Typed linear table evaluation; small original tables stay float64.

        'error' matches frozen PWTransformation/NIST interp1d after original
        clipping. 'clamp' matches TitrationFlash np.interp endpoint semantics.
        The interval uses the selected-precision operand view of the original
        table; the bound float64 source table is never changed or refitted.
        """
        import numpy as np
        self.require_pixels(signal)
        x = self._table(supports, name="linear supports")
        y = self._table(values, name="linear values")
        if x.ndim != 1 or x.shape != y.shape or len(x) < 2 or np.any(np.diff(x) <= 0):
            raise ValueError("Exact strictly ordered frozen interpolation table required")
        if endpoint not in ("error", "clamp"):
            raise ValueError("Explicit native endpoint rule required")
        xwork, ywork = self.pixels(x), self.pixels(y)
        if np.any(np.diff(xwork) <= 0):
            raise ValueError("Frozen interpolation supports collapse in selected precision")
        if endpoint == "error" and (np.any(signal < xwork[0]) or np.any(signal > xwork[-1])):
            raise ValueError("Signal exceeds the unchanged frozen interpolation bounds")
        indices = np.searchsorted(xwork, signal, side="left").clip(1, len(x)-1)
        lo, hi = indices-1, indices
        xlo, xhi = xwork[lo], xwork[hi]
        ylo, yhi = ywork[lo], ywork[hi]
        result = ((yhi-ylo)/(xhi-xlo))*(signal-xlo)+ylo
        if endpoint == "clamp":
            result = np.where(signal <= xwork[0], ywork[0], result)
            result = np.where(signal >= xwork[-1], ywork[-1], result)
        return result

    def titration(self, signal, flash):
        """Same frozen flash window/blend/LUT, with typed pixel arithmetic."""
        import numpy as np
        self.require_pixels(signal)
        if getattr(flash, "restoration", None) is not None:
            raise ValueError("Unimplemented restoration cannot be skipped by a precision variant")
        denominator = (flash.max_value_aq-flash.min_value_aq) or 1.0
        yellow = np.clip((signal-self.dtype(flash.min_value_aq)) /
                         self.dtype(denominator), self.dtype(0), self.dtype(1))
        physical = self.piecewise_linear(yellow, flash._lut_y, flash._lut_caq,
                                        endpoint="clamp")
        aqueous = self.dtype(1.0-flash.aqueous_blend)*physical + self.dtype(flash.aqueous_blend)*yellow
        gas = (np.clip(signal, self.dtype(flash.min_value_g), self.dtype(flash.max_value_g))
               - self.dtype(flash.min_value_g)) / self.dtype(flash.max_value_g-flash.min_value_g)
        return aqueous, gas

    def hydrostatic_fields(self, *, temperature, height, top_pressure,
                           water_density_20, water_density_23, density_20,
                           density_23, solubility_20, solubility_23):
        """Original CO2MassAnalysis formula/tables; no new physical parameters."""
        temp, height, top = map(self.pixels, (temperature, height, top_pressure))
        water = (self.dtype(water_density_20)*(self.dtype(23)-temp)/self.dtype(3)
                 + self.dtype(water_density_23)*(temp-self.dtype(20))/self.dtype(3))
        pressure = top + water*self.dtype(9.81)*height*self.dtype(1e-5)
        d20 = self.piecewise_linear(pressure, *density_20)
        d23 = self.piecewise_linear(pressure, *density_23)
        s20 = self.piecewise_linear(pressure, *solubility_20)
        s23 = self.piecewise_linear(pressure, *solubility_23)
        density = d20*(self.dtype(23)-temp)/self.dtype(3) + d23*(temp-self.dtype(20))/self.dtype(3)
        solubility = s20*(self.dtype(23)-temp)/self.dtype(3) + s23*(temp-self.dtype(20))/self.dtype(3)
        return dict(pressure=pressure, density=density, solubility=solubility)

    def selected_n1_fields(self, prepared, *, gain, factor, swi, aqueous_known,
                           gas_known, solubility, gas_density):
        """Use an exact faster optional device; retain the original CPU fallback."""
        self.require_pixels(prepared)
        if aqueous_known.dtype != bool or gas_known.dtype != bool:
            raise TypeError("Existing native boolean supports required")
        values = dict(gain=gain, factor=factor, swi=swi, aqueous_known=aqueous_known,
                      gas_known=gas_known, solubility=solubility, gas_density=gas_density)
        control = self._opencl.get("control")
        if control is None or control["mode"] == "cpu":
            return self._selected_n1_fields_cpu(prepared, **values)
        with self._opencl_lock:
            if "selector" not in self._opencl:
                from .mass_opencl import OptionalN1
                policy = self._opencl.get("policy")
                devices = None if policy is None else policy["qualified_devices"]
                self._opencl["selector"] = OptionalN1(self.precision, control, qualified_devices=devices,
                    verify_every_call=self._opencl.get("compare_cpu", False))
            selector = self._opencl["selector"]
        return selector.compute(self._selected_n1_fields_cpu, prepared, **values)

    def _selected_n1_fields_cpu(self, prepared, *, gain, factor, swi, aqueous_known,
                               gas_known, solubility, gas_density):
        """Selected frozen N1 only; original operation order and separate casts."""
        import numpy as np
        self.require_pixels(prepared)
        swi = self.pixels(swi)
        sol = self.pixels(solubility)
        rho = self.pixels(gas_density)
        if aqueous_known.dtype != bool or gas_known.dtype != bool:
            raise TypeError("Existing native boolean supports required")
        fraction = np.clip(prepared*self.dtype(gain), self.dtype(0), self.dtype(1))*self.dtype(factor)
        aqueous = np.where(aqueous_known, fraction, self.dtype(0))
        gas = np.zeros(prepared.shape, dtype=self.dtype)
        gas[gas_known] = self.dtype(1)-swi[gas_known]
        aqueous[gas_known] = self.dtype(1)
        raw_gas = rho*gas
        raw_aqueous = sol*aqueous*np.clip(self.dtype(1)-gas, self.dtype(0), None)
        # total is formed from raw operands, never from separately rounded fields.
        total = np.asarray(raw_gas+raw_aqueous, dtype=self.dtype)
        mass_aqueous = np.asarray(raw_aqueous, dtype=self.dtype)
        mass_gas = np.asarray(raw_gas, dtype=self.dtype)
        return dict(aqueous=aqueous, gas=gas, concentration=aqueous*sol,
                    mass_total=total, mass_aqueous=mass_aqueous, mass_gas=mass_gas)

    @staticmethod
    def integrate(value, weight, mask, *, space_dim=2):
        """Float64 weights/product/sums, original repeated axis-zero order."""
        import numpy as np
        if mask.dtype != bool or mask.shape != value.shape[:space_dim]:
            raise ValueError("Original native domain mask required")
        selected = np.where(mask, value, 0.0)
        weighted = np.asarray(weight, dtype=np.float64)*np.asarray(selected, dtype=np.float64)
        for _ in range(space_dim):
            weighted = np.sum(weighted, axis=0, dtype=np.float64)
        return weighted
