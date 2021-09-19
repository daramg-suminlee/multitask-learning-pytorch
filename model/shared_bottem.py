
import torch.nn as nn
from model.mlp import SingleLayerPerception as SLP
from model.mlp import MultiLayerPerceptron as MLP

class SharedBottom(nn.Module):
    def __init__(
        self,
        input_size:  int,
        shared_size: int,
        tower_size: int,
        num_tasks: int
    ):
        super(SharedBottom, self).__init__()
        self.num_tasks = num_tasks

        self.shared_layer = SLP(input_size, shared_size)
        self.tower_layer = nn.ModuleList(
            [MLP(shared_size, tower_size) for _ in range(num_tasks)]
        )

    def forward(self, x):
        shared = self.shared_layer(x)
        outs = [self.tower_layer[i](shared) for i in range(self.num_tasks)]
        return outs 
