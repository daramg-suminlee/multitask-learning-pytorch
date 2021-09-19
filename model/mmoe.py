
import torch
import torch.nn as nn
from model.mlp import SingleLayerPerception as SLP
from model.mlp import MultiLayerPerceptron as MLP

class MultigateMixtureOfExperts(nn.Module):
    def __init__(
        self,
        input_size: int,
        expert_size: int,
        tower_size: int,
        num_tasks: int,
        num_experts: int
    ):
        super(MultigateMixtureOfExperts, self).__init__()
        self.expert_size = expert_size
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        self.shared_layer = nn.ModuleList(
            [SLP(input_size, expert_size) for _ in range(num_experts)]
        )
        self.task_gate = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True) \
                for _ in range(num_tasks)]
        )
        self.tower_layer = nn.ModuleList(
            [MLP(expert_size, tower_size) for _ in range(num_tasks)]
        )

    def forward(self, x):
        task_gates = [torch.matmul(x, gate) for gate in self.task_gate]
        experts = [self.shared_layer[i](x) for i in range(self.num_experts)]
        shared_infos = []
        for t in range(self.num_tasks):
            gates = task_gates[t]
            shared_info = torch.zeros(gates.size()[0], self.expert_size)
            for i in range(self.expert_size):
                tmp = 0
                for j in range(self.num_experts):
                    tmp += gates[:,j] * experts[j][:,i]
                shared_info[:,i] = tmp
            shared_infos.append(shared_info)
        outs = [self.tower_layer[i](shared_infos[i]) for i in range(self.num_tasks)]
        return outs
