
import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, num_data, feature_dim, task_corr=0.9, scale=0.5, sin_param=10, seed=1):
        self.num_data = num_data
        self.feature_dim = feature_dim
        self.task_corr = task_corr
        torch.manual_seed(seed)

        # generate two orthogonal unit vectors u1 and u2
        u1, u2 = torch.rand(feature_dim), torch.rand(feature_dim)
        u1 -= u1.dot(u2) * u2 / torch.linalg.norm(u2)**2
        u1 /= torch.linalg.norm(u1)
        u2 /= torch.linalg.norm(u2)

        # generate two weight vector w1 and w2
        w1 = scale * u1
        w2 = scale * (task_corr*u1 + np.sqrt((1-task_corr**2))*u2)

        # randomly sample an input data point
        self.X = torch.normal(0, 1, size=(num_data, feature_dim))

        # generate two labels y1 and y2 for two tasks
        eps1, eps2 = np.random.normal(0, 0.01), np.random.normal(0, 0.01)
        sum1, sum2 = 0, 0
        for i in range(sin_param):
            alpha, beta = np.random.normal(0, 0.01), np.random.normal(0, 0.01)
            sum1 += torch.sin(alpha*torch.matmul(self.X, w1) + beta)
            sum2 += torch.sin(alpha*torch.matmul(self.X, w2) + beta)
        self.y1 = torch.matmul(self.X, w1) + sum1 + eps1
        self.y2 = torch.matmul(self.X, w2) + sum1 + eps2
        self.y = torch.transpose(
            torch.reshape(torch.cat((self.y1, self.y2)), (2, -1)), -1, 0
        )

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        X = self.X[index]
        y1 = self.y1[index]
        y2 = self.y2[index]
        return X, (y1, y2)