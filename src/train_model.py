import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

def train_model():
    x = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
    y = np.sin(x)
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(1000):
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), "model.pt")
    print("✅ Model trained and saved to model.pt")

if __name__ == "__main__":
    train_model()
