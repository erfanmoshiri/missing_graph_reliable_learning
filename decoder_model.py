import torch
import torch.nn as nn
from typing import List, Optional, Union, Callable

class MLPDecoder(nn.Module):
    """
    Dynamic MLP decoder.
    Args:
        in_dim: input feature dimension.
        layer_dims: list of layer output sizes (last one is final output dimension).
        activation: activation for hidden layers (default ReLU).
        out_activation: optional activation applied only to final layer output.
        dropout: dropout applied after each hidden layer (before next linear).
        batch_norm: if True, applies BatchNorm1d after each hidden Linear (before activation).
    """
    def __init__(
        self,
        in_dim: int,
        layer_dims: List[int],
        activation: Union[str] = "relu",
        out_activation: Optional[Union[str, Callable[[], nn.Module]]] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        if len(layer_dims) == 0:
            raise ValueError("layer_dims must contain at least one dimension (output).")

        act_fn = self._resolve_activation(activation)
        out_act_fn = self._resolve_activation(out_activation) if out_activation else None

        layers = []
        prev = in_dim
        for i, dim in enumerate(layer_dims):
            is_last = i == len(layer_dims) - 1
            layers.append(nn.Linear(prev, dim))
            if not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dim))
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            prev = dim

        self.net = nn.Sequential(*layers)
        self.out_activation = out_act_fn() if out_act_fn else None
        self.output_dim = layer_dims[-1]

    def _resolve_activation(self, act) -> Callable[[], nn.Module]:
        if act is None:
            return lambda: nn.Identity()
        lookup = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "selu": nn.SELU,
            "identity": nn.Identity,
        }
        if act not in lookup:
            raise ValueError(f"Unknown activation: {act}")
        return lookup[act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

