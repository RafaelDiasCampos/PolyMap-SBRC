import torch
import torch.nn as nn
import torch.nn.init as init
import math


def lecun_uniform_(tensor):
    fan_in, _ = init._calculate_fan_in_and_fan_out(tensor)
    bound = math.sqrt(3.0 / fan_in)
    return init.uniform_(tensor, -bound, bound)


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_hidden_layers: int,
        n_neurons: int,
        dropout_prob: float = 0.5,
        dropout_function: nn.Module = nn.Dropout,
        activation_function: nn.Module = nn.ReLU,
        initialization_function: callable = nn.init.xavier_uniform_,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.dropout_prob = dropout_prob

        self.dropout_function = dropout_function
        self.activation_function = activation_function
        self.initialization_function = initialization_function

        self.model = self._build_model()
        self._init_weights()

    def _build_model(
        self,
    ) -> nn.Sequential:
        layers = []

        # Initial linear layer
        layers.append(nn.Linear(self.in_features, self.n_neurons))
        layers.append(self.activation_function())

        # Hidden layers with activation and dropout
        for _ in range(self.n_hidden_layers):
            layers.append(nn.Linear(self.n_neurons, self.n_neurons))
            layers.append(self.activation_function())
            layers.append(self.dropout_function(self.dropout_prob))

        # Final linear + log-softmax
        layers.append(nn.Linear(self.n_neurons, self.out_features))
        layers.append(nn.LogSoftmax(dim=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        # Apply Xavier (Glorot) uniform init to all Linear layers
        def init_fn(m):
            if isinstance(m, nn.Linear):
                self.initialization_function(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.model.apply(init_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FNN(NeuralNetwork):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_hidden_layers: int,
        n_neurons: int,
        dropout_prob: float = 0.3,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            n_hidden_layers=n_hidden_layers,
            n_neurons=n_neurons,
            dropout_prob=dropout_prob,
            dropout_function=nn.Dropout,
            activation_function=nn.ReLU,
            initialization_function=nn.init.xavier_uniform_
        )


class SNN(NeuralNetwork):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_hidden_layers: int,
        n_neurons: int,
        dropout_prob: float = 0.1,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            n_hidden_layers=n_hidden_layers,
            n_neurons=n_neurons,
            dropout_prob=dropout_prob,
            dropout_function=nn.AlphaDropout,
            activation_function=nn.SELU,
            initialization_function=lecun_uniform_,
        )
