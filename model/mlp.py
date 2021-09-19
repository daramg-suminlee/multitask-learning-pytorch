
import torch.nn as nn

class SingleLayerPerception(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(SingleLayerPerception, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LogSoftmax(1)
        )
    
    def forward(self, x):
        return self.layer(x)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(MultiLayerPerceptron, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.LogSigmoid()
        )

    def forward(self, x):
        return self.layer(x)
