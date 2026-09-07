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

    def initialize_random(self, low=-1, high=1, seed=None):
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

    def train(self, learning_rate, training_data, epochs):
        if not self.layers:
            raise RuntimeError(
                "Network has no layers. Call initialize_random() before training."
            )
        if len(training_data) == 0:
            raise ValueError("training_data must contain at least one sample.")

        bar_width = 30
        update_every = max(1, len(training_data) // 100)

        for epoch in range(epochs):
            epoch_loss = 0.0
            print(f"\nEpoch {epoch + 1}/{epochs}")

            for sample_index, sample in enumerate(training_data):
                network_input = sample[0]
                result = self.forward(network_input)
                target = np.atleast_1d(np.asarray(sample[1], dtype=float))
                if target.shape != result.shape:
                    raise ValueError(f"Expected target shape {result.shape}, got {target.shape}.")
                error = result - target
                loss = 0.5 * float(np.sum(error ** 2))
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


                for layer in self.layers:
                    for neuron in layer.neurons:
                        weight_gradients = neuron.delta * neuron.last_inputs
                        neuron.weights -= learning_rate * weight_gradients
                        neuron.bias -= learning_rate * neuron.delta

            average_loss = epoch_loss / len(training_data)
            print(f"\r  [{'#' * bar_width}] 100.0% | average loss: {average_loss:.6f}")
