# train.py

import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from model import GNN

def train():
    dataset = QM9(root='data/QM9')

    target_idx = 10

    train_dataset = dataset[:10000]
    test_dataset = dataset[10000:12000]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(device)

            pred = model(data.x.float(), data.edge_index, data.batch)
            y = data.y[:, target_idx].view(-1, 1)

            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train()
