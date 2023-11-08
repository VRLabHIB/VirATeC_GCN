# GCN_Viratec
## Idea
Training graphical neural networks on gaze transitions and fixation information. 

## Links and Literature

**Paper using GCN on gaze data (gesture):** https://www.frontiersin.org/articles/10.3389/frobt.2021.709952/full#B10

**Model based on this arxiv article:** https://arxiv.org/pdf/1609.02907.pdf

- *with the GitHub:* https://antonsruberts.github.io/graph/gcn/


**Articles for basic GCN implementation**

- https://github.com/tkipf/gcn/blob/master/gcn/models.py


- ID 006 had to be removed, because we only had data of half the session


## Structure:

- Goal: Predict complexity of the teaching situation (atm predict expertise level)

- Data: VirATeC (one sample is 30 seconds interval)

- Convert classroom AOIs into graph:
	- Each virtual student + board is a Node
	- Egdes are gaze transitions between the nodes
	
- Graph Aggregation: Each graph (30 sec) consists of
	- Node Features: V(n_nodes, n_features)
		- AOI duration
		- Mean pupil diameter (substractive baseline corrected)
	- Edge Index: V(n_edges, 1)
		- Binary value: two nodes are connected=1 or not connected=0, if gaze transition occured during interval between two nodes
	- Edge Attributes: V(n_edges, n_attributes)
		- Edge weights: frequency of gaze transitions
		- Transition duration: average time to transitoin from one node to another

- Model: Graph Convolution Network
	- 1-3 layers of (GCN -> graph normalization -> relu) -> global pooling layer -> fully connected with 2-dim output -> softmax
	- Parameters:
		- train_test_split = 0.8
		- batch_size = 50
		- hidden_channels = 50
		- learning_rate = 0.001
		- nepoch = 200
