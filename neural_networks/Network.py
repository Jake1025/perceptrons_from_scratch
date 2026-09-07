import numpy as np
from Layer import Layer
from Activations import Activation


class Network:
    def __init__(self, structure: list[tuple[int, Activation | None]]):
        """The first tuple specifies raw inputs; its activation is ignored."""
        if len(structure) < 2:
            raise ValueError("Provide an input specification and at least one layer.")
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0
               for size, _ in structure):
            raise ValueError("All sizes must be positive integers.")
        if any(not isinstance(activation, Activation) for _, activation in structure[1:]):
            raise ValueError("Each trainable layer requires an Activation.")
        self.input_size = structure[0][0]
        self.layer_specs = structure[1:]
        self.layers = []

    def initialize_random(self, low=None, high=None, seed=None):
        rng = np.random.default_rng(seed)
        self.layers = []

        input_size = self.input_size

        for neuron_count, activation in self.layer_specs:
            layer = Layer(
                input_size=input_size,
                neuron_count=neuron_count,
                activation=activation,
                rng=rng,
                low=low,
                high=high,
            )

            self.layers.append(layer)
            input_size = neuron_count

    def forward(self, inputs):
        if not self.layers:
            raise RuntimeError(
                "Network has no layers. Call initialize_random() before forward() or train()."
            )

        outputs = np.asarray(inputs, dtype=float)
        if outputs.shape != (self.input_size,):
            raise ValueError(f"Expected input shape {(self.input_size,)}, got {outputs.shape}.")

        for layer in self.layers:
            outputs = layer.forward(outputs)

        return outputs

    def train(self, learning_rate, training_data, epochs, *,
              patience=2, min_improvement=0.01, decay=0.5,
              min_learning_rate=1e-6, max_gradient_norm=1.0,
              on_epoch_end=None, shuffle=True, seed=None):
        """Clip large sample gradients and decay the rate on loss plateaus.

        min_improvement is a relative fraction; patience=None disables decay.
        max_gradient_norm=None disables clipping. Scheduling resets per call.
        Returns per-epoch loss and the learning rate used for that epoch.
        on_epoch_end(epoch, metrics) can report or add metrics after updates.
        Shuffle sample indices each epoch without modifying training_data.
        seed makes the shuffled order reproducible; shuffle=False keeps order.
        """
        if not self.layers:
            raise RuntimeError(
                "Network has no layers. Call initialize_random() before training."
            )
        if len(training_data) == 0:
            raise ValueError("training_data must contain at least one sample.")
        if learning_rate <= 0 or min_learning_rate <= 0 or not 0 < decay < 1:
            raise ValueError("Rates must be positive and decay must be between 0 and 1.")
        if (patience is not None and patience < 1) or not 0 <= min_improvement < 1:
            raise ValueError("Use positive patience and a relative improvement in [0, 1).")
        if max_gradient_norm is not None and max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive or None.")

        bar_width = 30
        update_every = max(1, len(training_data) // 100)
        previous_loss = None
        stalled_epochs = 0
        history = []
        rng = np.random.default_rng(seed)

        for epoch in range(epochs):
            epoch_loss = 0.0
            print(f"\nEpoch {epoch + 1}/{epochs}")

            order = rng.permutation(len(training_data)) if shuffle else range(len(training_data))
            for sample_index, data_index in enumerate(order):
                sample = training_data[data_index]
                network_input = sample[0]
                result = self.forward(network_input)
                target = np.atleast_1d(np.asarray(sample[1], dtype=float))
                if target.shape != result.shape:
                    raise ValueError(f"Expected target shape {result.shape}, got {target.shape}.")
                error = result - target
                loss = 0.5 * float(np.sum(error ** 2))
                if not np.isfinite(loss):
                    raise FloatingPointError("Non-finite loss: restart with fresh weights and a lower rate.")
                epoch_loss += loss

                if sample_index % update_every == 0 or sample_index == len(training_data) - 1:
                    progress = (sample_index + 1) / len(training_data)
                    filled = int(bar_width * progress)
                    bar = '#' * filled + '-' * (bar_width - filled)
                    print(f"\r  [{bar}] {progress:6.1%}", end="", flush=True)


                for l_idx, layer in reversed(list(enumerate(self.layers))):
                    for n_idx, neuron in enumerate(layer.neurons):

                        derivative = neuron.activation_derivative()

                        if l_idx == len(self.layers) - 1:
                            delta = error[n_idx] * derivative
                        else:
                            next_layer = self.layers[l_idx + 1]

                            propagated_error = sum(
                                next_neuron.delta * next_neuron.weights[n_idx]
                                for next_neuron in next_layer.neurons
                            )

                            delta = propagated_error * derivative

                        neuron.delta = delta


                gradients = [
                    (neuron, neuron.delta * neuron.last_inputs, neuron.delta)
                    for layer in self.layers for neuron in layer.neurons
                ]
                # One scale for ALL weights and biases preserves the gradient direction.
                norm = np.linalg.norm(np.concatenate([
                    np.append(weights, bias) for _, weights, bias in gradients
                ]))
                if not np.isfinite(norm):
                    raise FloatingPointError("Non-finite gradient: restart with fresh weights and a lower rate.")
                scale = 1.0 if max_gradient_norm is None else min(
                    1.0, max_gradient_norm / max(norm, 1e-12)
                )
                for neuron, weight_gradient, bias_gradient in gradients:
                    neuron.weights -= learning_rate * scale * weight_gradient
                    neuron.bias -= learning_rate * scale * bias_gradient

            average_loss = epoch_loss / len(training_data)
            history.append({"loss": average_loss, "learning_rate": learning_rate})
            print(f"\r  [{'#' * bar_width}] 100.0% | average loss: {average_loss:.6f} | lr: {learning_rate:g}")
            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, history[-1])
            if patience is not None and previous_loss is not None:
                improvement = (previous_loss - average_loss) / max(previous_loss, 1e-12)
                stalled_epochs = stalled_epochs + 1 if improvement < min_improvement else 0
                if stalled_epochs >= patience:
                    learning_rate = min(learning_rate, max(min_learning_rate, learning_rate * decay))
                    stalled_epochs = 0
            previous_loss = average_loss

        return history
