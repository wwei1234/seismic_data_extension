from dataclasses import dataclass

from config import (
    F3_MAX_LEAKAGE,
    F3_MIN_CORRELATION,
    F3_MIN_PHASE,
    SYNTHETIC_TIE_TOLERANCE,
)


def domain_cycle(epoch):
    epoch = int(epoch)
    if epoch <= 60:
        return ("f3",)
    if epoch <= 180:
        return ("f3", "f3", "synthetic")
    return ("f3", "synthetic")


@dataclass
class GatedCheckpointSelector:
    best_epoch: int | None = None
    best_f3_correlation: float = float("-inf")
    best_synthetic_correlation: float = float("-inf")
    best_f3: dict | None = None
    best_synthetic: dict | None = None

    @staticmethod
    def passes_gate(f3):
        return (
            float(f3["correlation"]) >= F3_MIN_CORRELATION
            and float(f3["phase"]) >= F3_MIN_PHASE
            and float(f3["leakage"]) <= F3_MAX_LEAKAGE
        )

    def consider(self, epoch, f3, synthetic):
        if not self.passes_gate(f3):
            return False
        synthetic_correlation = float(synthetic["residual_correlation"])
        f3_correlation = float(f3["correlation"])
        if self.best_epoch is None:
            choose = True
        else:
            difference = synthetic_correlation - self.best_synthetic_correlation
            choose = (
                difference > SYNTHETIC_TIE_TOLERANCE
                or (
                    abs(difference) <= SYNTHETIC_TIE_TOLERANCE
                    and f3_correlation > self.best_f3_correlation
                )
            )
        if not choose:
            return False
        self.best_epoch = int(epoch)
        self.best_f3_correlation = f3_correlation
        self.best_synthetic_correlation = synthetic_correlation
        self.best_f3 = dict(f3)
        self.best_synthetic = dict(synthetic)
        return True
