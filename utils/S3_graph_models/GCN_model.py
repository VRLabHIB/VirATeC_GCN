import torch
import random
from IPython.core.display_functions import display
from torch_geometric.datasets import TUDataset
import numpy as np
import pandas as pd
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Softmax
from torch_geometric.nn import GCNConv, GraphConv, TAGConv, GraphNorm, PairNorm
from torch_geometric.nn import global_mean_pool, global_add_pool, PANPooling, global_max_pool
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.pyplot as plt


def run_GCN_model(dataset):
    train_test_split = 0.8
    batch_size = 5
    hidden_channels = 10
    learning_rate = 0.01
    nepoch = 200

    # torch.manual_seed(12345)
    # random.seed(12345)

    random.shuffle(dataset)
    print('len of dataset {}', len(dataset))

    labels = list()
    for data in dataset:
        labels.append(data.y)
    df_labels = pd.DataFrame({'index': np.arange(len(dataset)), 'label':labels})
    n_label_1 = len(df_labels[df_labels['label'] == 1])
    n_label_0 = len(df_labels[df_labels['label'] == 0])

    ratio = n_label_1/len(dataset)
    train_dataset = dataset[:int(train_test_split * len(dataset))]
    test_dataset = dataset[int(train_test_split * len(dataset)):]


    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print(' ')

    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(' ')
    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            torch.manual_seed(12345)
            num_node_features = 4

            #self.batch_norm = BatchNorm1d(hidden_channels) #, affine=False)
            self.conv1 = GCNConv(num_node_features, hidden_channels)  # 1= num node features
            self.norm1 = GraphNorm(hidden_channels)

            #self.conv2 = TAGConv(hidden_channels, hidden_channels)
            #self.norm2 = GraphNorm(hidden_channels)

            #self.conv3 = TAGConv(hidden_channels*2, hidden_channels)
            #self.norm3 = PairNorm(hidden_channels)

            self.lin = Linear(hidden_channels, 2)  # 2= num classes
            self.soft = Softmax(dim=1)

        def forward(self, x, edge_index, edge_weight, batch):
            # 1. Obtain node embeddings
            x = self.conv1(x, edge_index, edge_weight)
            x = self.norm1(x)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_weight)
            x = x.relu()
            #x = self.norm2(x)

            #x = self.conv3(x, edge_index, edge_weight)
            #x = self.norm3(x)
            #x = x.relu()

            # 2. Readout layer
            x = global_add_pool(x, batch)  # [batch_size, hidden_channels]
            x = x.relu()


            # 3. Apply a final classifier
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin(x)
            x = self.soft(x)

            return x

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print(' ')

        #model = GCN(hidden_channels=hidden_channels)
        #print(model)
        #print(' ')

        from IPython.display import Javascript

        display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

        model = GCN(hidden_channels=hidden_channels)
        print(model)
        print(' ')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        def train():
            model.train()

            for data in train_loader:  # Iterate in batches over the training dataset.
                #data.edge_index, data.edge_weight = gcn_norm(data.edge_index, data.edge_weight)
                out = model(data.x, data.edge_index, data.edge_weight, data.batch)  # Perform a single forward pass.
                y_t = data.y
                y_r = (y_t+y_t) % 2
                y = torch.zeros(len(data.y), 2)
                y[:, 0] = y_t
                y[:, 1] == y_r
                loss = criterion(out, data.y)  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.

        def test(loader, used_for, epoch):
            model.eval()
            from sklearn.metrics import f1_score

            correct = 0
            pred_lst = list()
            true_lst = list()
            for data in loader:  # Iterate in batches over the training/test dataset.
                #data.edge_index, data.edge_weight = gcn_norm(data.edge_index, data.edge_weight)
                out = model(data.x, data.edge_index, data.edge_weight, data.batch)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                correct += int((pred == data.y).sum())  # Check against ground-truth labels.

                pred_lst += pred.tolist()
                true_lst += data.y.tolist()

            f1_score = f1_score(true_lst, pred_lst)
            return correct / len(loader.dataset), f1_score  # Derive ratio of correct predictions.

        for epoch in range(1, nepoch):
            train()
            train_acc, train_f1 = test(train_loader,'train_set', epoch)
            test_acc, test_f1 = test(test_loader, 'test_set', epoch)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            print(f'Epoch: {epoch:03d}, Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}')
            print('-----------')
