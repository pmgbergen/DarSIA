from dataclasses import dataclass, field


@dataclass
class BeckmannConvergenceHistory:
    """Class to store the convergence history."""

    distance: list[float] = field(default_factory=list)
    residual: list[float] = field(default_factory=list)
    increment: list[float] = field(default_factory=list)
    distance_increment: list[float] = field(default_factory=list)
    timings: list[dict] = field(default_factory=list)
    total_run_time: list[float] = field(default_factory=list)

    def append(
        self,
        distance: float,
        distance_increment: float,
        increment: float,
        residual: float,
        timings: dict,
        total_run_time: float,
    ) -> None:
        self.distance.append(distance)
        self.distance_increment.append(distance_increment)
        self.increment.append(increment)
        self.residual.append(residual)
        self.timings.append(timings)
        self.total_run_time.append(total_run_time)

    def as_dict(self) -> dict:
        return {
            "distance": self.distance,
            "distance_increment": self.distance_increment,
            "increment": self.increment,
            "residual": self.residual,
            "timings": self.timings,
            "total_run_time": self.total_run_time,
        }
