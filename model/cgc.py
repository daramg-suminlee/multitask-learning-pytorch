
import torch
import torch.nn as nn
from model.mlp import SingleLayerPerception as SLP
from model.mlp import MultiLayerPerceptron as MLP

class CustomizedGateControl(nn.Module):
    def __init__(
        self,
        input_size: int,
        expert_size: int,
        tower_size: int,
        num_tasks: int,
        num_shared_experts: int,
        num_task_experts: list
    ):
        super(CustomizedGateControl, self).__init__()
        self.expert_size = expert_size
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.num_tasks = num_tasks

        self.shared_expert_layer = nn.ModuleList(
            [SLP(input_size, expert_size) for _ in range(num_shared_experts)]
        )
        self.task_expert_layer = nn.ModuleList([nn.ModuleList(
            [SLP(input_size, expert_size) for _ in range(num_experts)]
        ) for num_experts in num_task_experts])
        self.task_gate = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_size, num_shared_experts + num_task_experts[i]), \
                requires_grad=True) for i in range(num_tasks)]
        )
        self.tower_layer = nn.ModuleList(
            [MLP(expert_size, tower_size) for _ in range(num_tasks)]
        )

    def forward(self, x):
        task_gates = [torch.matmul(x, gate) for gate in self.task_gate]
        shared_experts = [self.shared_expert_layer[i](x) for i in range(self.num_shared_experts)]
        shared_infos = []
        for t in range(self.num_tasks):
            num_task_experts = self.num_task_experts[t]
            task_expert_layer = self.task_expert_layer[t]
            task_experts = [task_expert_layer[i](x) for i in range(num_task_experts)]

            gates = task_gates[t]
            shared_info = torch.zeros(gates.size()[0], self.expert_size)
            for i in range(self.expert_size):
                tmp = 0
                for j in range(self.num_shared_experts):
                    tmp += gates[:,j] * shared_experts[j][:,i]
                for k in range(self.num_task_experts[t]):
                    tmp += gates[:,k+self.num_shared_experts] * task_experts[k][:,i]
                shared_info[:,i] = tmp
            shared_infos.append(shared_info)
        outs = [self.tower_layer[i](shared_infos[i]) for i in range(self.num_tasks)]
        return outs
