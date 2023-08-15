import torch
import random
from IPython.core.display_functions import display
from torch_geometric.datasets import TUDataset
import numpy as np
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data


def run_GCN_model(dataset):
    # TODO: different models can either use edge weights or edged attributes, but always need node features
    torch.manual_seed(12345)
    random.seed(12345)

    random.shuffle(dataset)

    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print(' ')

    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            torch.manual_seed(12345)
            self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lin = Linear(hidden_channels, dataset.num_classes)

        def forward(self, x, edge_index, edge_weight, batch):
            # 1. Obtain node embeddings
            x = self.conv1(x, edge_index, edge_weight)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)

            # 2. Readout layer
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

            # 3. Apply a final classifier
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin(x)

            return x

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print(' ')

        model = GCN(hidden_channels=64)
        print(model)
        print(' ')

        from IPython.display import Javascript

        display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

        model = GCN(hidden_channels=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        def train():
            model.train()

            for data in train_loader:  # Iterate in batches over the training dataset.
                out = model(data.x, data.edge_index, data.edge_weight, data.batch)  # Perform a single forward pass.
                loss = criterion(out, data.y)  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.


        def test(loader, epoch):
            model.eval()

            correct = 0
            pred_lst = list()
            true_lst = list()
            for data in loader:  # Iterate in batches over the training/test dataset.
                out = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                correct += int((pred == data.y).sum())  # Check against ground-truth labels.

                pred_lst.append(pred)
                true_lst.append(data.y)
            print(' ')
            return correct / len(loader.dataset)  # Derive ratio of correct predictions.


        for epoch in range(1, 171):
            train()
            train_acc = test(train_loader)
            test_acc = test(test_loader, epoch)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')