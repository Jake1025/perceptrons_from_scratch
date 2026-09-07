from enum import Enum
import math

class Activation(Enum):
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    LINEAR = "linear"

    def weight_limit(self, input_size: int, output_size: int) -> float:
        """Uniform initialization: He for ReLU, Xavier for other activations."""
        if self is Activation.RELU:
            return math.sqrt(6.0 / input_size)
        return math.sqrt(6.0 / (input_size + output_size))

    def apply(self, value: float) -> float:
        if self is Activation.SIGMOID:
            if value >= 0:
                return 1 / (1 + math.exp(-value))
            exp_value = math.exp(value)
            return exp_value / (1 + exp_value)

        if self is Activation.RELU:
            return max(0.0, value)

        if self is Activation.TANH:
            return math.tanh(value)

        if self is Activation.LINEAR:
            return value

        raise ValueError(f"Unsupported activation: {self}")

    def derivative(self, weighted_sum: float, activation: float) -> float:
        if self is Activation.SIGMOID:
            return activation * (1 - activation)

        if self is Activation.RELU:
            return 1.0 if weighted_sum > 0 else 0.0

        if self is Activation.TANH:
            return 1 - activation ** 2

        if self is Activation.LINEAR:
            return 1.0

        raise ValueError(f"Unsupported activation: {self}")
