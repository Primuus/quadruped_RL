"""Shared simulation-only Jenkins friction law (NumPy/Torch via clip adapter)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class JointDynamicsParameters:
    joint_names: tuple
    coulomb_torque: float
    viscous_damping: float
    presliding_stiffness: float

    @classmethod
    def from_json(cls, path):
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        expected = {"version", "model", "joint_names", "coulomb_torque",
                    "viscous_damping", "presliding_stiffness"}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("Joint dynamics must use the complete version 1 schema")
        if type(raw["version"]) is not int or raw["version"] != 1 or raw["model"] != "jenkins":
            raise ValueError("Unsupported joint dynamics version or model")
        names = raw["joint_names"]
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("joint_names must be a nonempty list of unique names")
        values = {}
        for key in ("coulomb_torque", "viscous_damping", "presliding_stiffness"):
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(key + " must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(key + " must be finite and positive")
            values[key] = float(value)
        return cls(joint_names=tuple(names), **values)

    def validate_joint_names(self, names):
        if tuple(names) != self.joint_names:
            raise ValueError("Joint dynamics order does not match the loaded robot")


class JenkinsJointFriction:
    """Elastic presliding followed by bounded dry slip, plus viscous damping.

    z is angular bristle deflection [rad], k*z is torque [N m].
    Only measured displacement advances z; reset removes history after teleport.
    Spring energy k*z**2/2 may be returned on reversal: instantaneous power
    need not always be negative. This is not a motor compensation controller.
    """

    def __init__(self, parameters, position, clip):
        self.parameters = parameters
        self.clip = clip
        self.yield_deflection = parameters.coulomb_torque / parameters.presliding_stiffness
        self.previous_position = position * 1.0
        self.deflection = position * 0.0

    def reset(self, position, env_ids=None):
        selected = slice(None) if env_ids is None else env_ids
        self.previous_position[selected] = position[selected]
        self.deflection[selected] = 0.0

    def compute(self, position, velocity):
        self.deflection[...] = self.clip(
            self.deflection + (position - self.previous_position),
            -self.yield_deflection, self.yield_deflection,
        )
        self.previous_position[...] = position
        return (-self.parameters.presliding_stiffness * self.deflection
                - self.parameters.viscous_damping * velocity)
