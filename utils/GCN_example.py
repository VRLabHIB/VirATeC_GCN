import torch
from IPython.core.display_functions import display
from torch_geometric.datasets import TUDataset
import numpy as np
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
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

if __name__ == '__main__':


    dataset = TUDataset(root='data/TUDataset', name='MUTAG')

    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    edge_weights = np.random.rand(38)

    data1 = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, edge_weights=edge_weights, y=data.y)
    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    print(' ')

    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print(' ')

    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
                out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
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

