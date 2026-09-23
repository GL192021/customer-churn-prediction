import torch

class LogisticRegression_torch(torch.nn.Module):
    def __init__(self, x_dim, y_dim=1):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.model = torch.nn.Linear(x_dim, y_dim)

    def forward(self, x):
        y_pred_logit = self.model(x)
        return y_pred_logit