"""Canonical SHA-bound optional N1 control, safe before scientific imports."""

FORMAT = "darsia-opencl-n1-v1"


def optional_n1_configuration(value):
    """Read the control from SHA-bound run DATA, never an environment override."""
    if value is None:
        return dict(format=FORMAT, mode="cpu", device_id=None)
    if (not isinstance(value, dict) or set(value) - {"format", "mode", "device_id"}
            or value.get("format") != FORMAT or value.get("mode") not in {"auto", "cpu"}
            or (value.get("device_id") is not None
                and (type(value["device_id"]) is not int or value["device_id"] < 0))):
        raise ValueError("Explicit optional N1 OpenCL control required")
    return dict(format=FORMAT, mode=value["mode"], device_id=value.get("device_id"))
