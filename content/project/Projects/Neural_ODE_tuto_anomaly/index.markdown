---
title: "Fraud detection with Graph Attention Networks"
subtitle: ""
date: 2022-12-01
author: "Nadhir Hassen"
draft: false
tags:
  - Graphical neural network
  - Anomaly detection
  - Dynamical System
  - Generative model
categories:
  - Graphical neural network
  - Anomaly detection
  - Dynamical System
  - Generative model
layout: single
links:
- icon: box
  icon_pack: fas
  name: Pytorch Geometric Doc
  url: https://pytorch-geometric.readthedocs.io/en/latest/index.html
# - icon: comment
#   icon_pack: fas
#   name: related talk
#   url: "https://github.com/SciML/DiffEqFlux.jl"
# - icon: chart-bar
#   icon_pack: fas
#   name: tidytuesday databases on notion
#   url: tiny.cc/notion-dataviz
---


## Introduction to Fraud detection 

Fraud detection is a set of processes and analyses that allow businesses to identify and prevent unauthorized financial activity. This can include fraudulent credit card transactions, identify theft, cyber hacking, insurance scams, and more. A fraud occurs when someone takes money or other assets from you through deception or criminal activity.

Consequently, having an effective fraud detection system can help institutions to identify suspicious behaviors or accounts and minimize losses if the fraud is ongoing.The fraud detection model based on ML algorithms is challenging for many reasons: fraud represent a small portion of all the daily transactions, its distribution evolves quickly over time then true transaction label will be only available after several days, because investigators could not timely check all the transactions.

But traditional methods of Machine learning still fail to detect a fraud because most data science models omit something critically important: network structure.

Fraud detection like social networks imply the use of the power of a Graph. The following figure is an example of graph transactions network, we can see some nodes like bank account, credit card, person with their relationships.


In fact, tabular data models, with data organized in rows and columns, are not designed for capturing the complex relationships and network structure inherent in your data. Analyzing data as a graph enables us to reveal and use its structure for predictions.

Therefore, using graphs handles many complex issues, it works for huge amount of data with multiple edges relationships that can change over time. Graph Machine learning improves the accuracy of fraud predictions by using more relevant features information from the network.


# Graph Representation

A node in the graph represents a transaction, an edge represents a flow of Bitcoins between one transaction and the other. Each node has 166 features and has been labeled as being created by a “licit”, “illicit” or “unknown” entity.

There are 203,769 nodes and 234,355 edges in the graph, we have two percent (4,545) of the nodes that are labelled class1 (illicit), and twenty-one percent (42,019) are labelled class2 (licit) as we can see in Figure 2. The remaining transactions are not labelled with regard to licit versus illicit.There are 49 distinct time steps.


The first 94 features represent local information about the transaction and the remaining 72 features are aggregated features obtained using transaction information one-hop backward/forward from the center node — giving the maximum, minimum, standard deviation and correlation coefficients of the neighbor transactions for the same information data (number of inputs/outputs, transaction fee, etc.).

So with this dataset, we will work on a node classification task, where the goal is to predict if a node is licit or illicit.

A working knowledge of Pytorch is required to understand the programming examples.We will assume a basic understanding of machine learning, neural networks and backpropagation.

# Graph Neural Networks

The Graph Neural Networks (GNNs) [8,9,10] is gaining increasing popularity. GNNs are neural networks that can be directly applied to graphs and provide an easy way to do node-level, edge-level, and graph-level prediction tasks.

Recent years GNNs have seen significant developments and successes in many problems like the fields of biology, chemistry, social science, physics, and many others. It has led to state-of-the-art performance on several benchmarks. GNNs consider not only instance-level features but also graph-level features by performing message passage to agglomerate information from neighbors when making decisions, which is shown in the following figure. This leads to great performance on this kind of task.


<div class="figure" style="text-align: center">
<img src="img/graph.png" alt="The deploy contexts section after clicking the Edit settings button. This section shows three settings that can be edited. The first is the production branch which is set to 'main' in a free text box. The second is deploy previews which is a radio button set to 'any pull request against your production branch/branch deploy branches (as opposed to 'none'). The third is branch deploys which is a radio button set to 'all' (as opposed to 'none' and 'let me add individual branches'). There are two buttons at the bottom of this section, Save and Cancel." width="75%" />
<p class="caption">Figure 1: GNN Mechanism </p>
</div>


GNN converts a graphical relationship into a system where information messages are passed from neighborhood nodes through edges and aggregated into the target node (Figure 5 right). There are many variants of GNN which differ to each other on how each node aggregates and combines the representations of its neighbors with its own.

In this blog post, for a fraud detection task on the Bitcoin transactions, we will present you the attention mechanism through the original GAT model then we will show you a second version called GATv2.

Graph Attention Networks (GATs) are one of the most popular GNN architectures that performs better than other models on several benchmark and tasks, was introduced by Velickovic et al. (2018). Both these versions leverage the “Attention” mechanism [5] which has shown great success in various ML fields, e.g. for NLP with Transformers.

We will use the Pytorch Geometric PyG, which is the most popular graph deep learning framework built on top of Pytorch. PyG is suitable to quickly implement GNN models, with abundant graph models already implemented for a wide range of applications related to structured data. Moreover, GraphGym, available as part of the PyG, allows a training/evaluation pipeline to be built in a few lines of code.

# Graph Attention Networks

GraphSAGE and many other popular GNN architectures weigh all neighbors messages with equal importance (e.g mean or max-pooling as AGGREGATE). However, every node in a GAT model updates its representation by attending to its neighbors using its own representation as the query. Thus, every node computes a weighted average of its neighbors, and selects its most relevant neighbors.

This model utilizes an attention mechanism α(ij) to determine the importance of each message being passed by different nodes in the neighborhood as in the following Figure, showing a single attention mechanism

To compute the attention score between two neighbors, a scoring function e computes a score for every edge h(j,i) which indicates the importance of the features of the neighbor j to the node i where a shared attentional mechanism “a” and a shared linear transformation parametrized by the weight matrix “W” are learned.


# Implementation of GAT with Python Geometric(PyG) 

We can implement a simplified version of a GAT conv layer using PyG, based on the equations listed above. Equations and dimensions of output of each layer have been commented to improve readability and understanding.

PyG provides the MessagePassing base class, which helps in creating such kinds of message passing graph neural networks by automatically taking care of message propagation. The base class provides a few helpful functions.

Firstly, message() function allows you to define what node information do we want to pass for each edge, and the aggregate function allows us to define how we intend to merge the messages from all edges to the target node (“add”, “mean”, “max”, etc). Finally, the propagate() function helps us to run the message passing and aggregation over all the edges and nodes in the graph. Further details can be found at the official PyG documentation [11].

Even though a prebuilt GATConv is available, let’s start by writing a custom GAT layer using the messagePassing base class to better understand how it works.


```python
class myGAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads = 2,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(myGAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels # node features input dimension
        self.out_channels = out_channels # node level output dimension
        self.heads = heads # No. of attention heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None
     
        # Initialization
        self.lin_l = Linear(in_channels, heads*out_channels)
        self.lin_r = self.lin_l

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels).float())
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels).float())

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):
        
        H, C = self.heads, self.out_channels # DIM：H, outC

        #Linearly transform node feature matrix.
        x_source = self.lin_l(x).view(-1,H,C) # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC]
        x_target = self.lin_r(x).view(-1,H,C) # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC]

        # Alphas will be used to calculate attention later
        alpha_l = (x_source * self.att_l).sum(dim=-1) # DIM: [nodes, H, outC] x [H, outC] => [nodes, H]
        alpha_r = (x_target * self.att_r).sum(dim=-1) # DIM: [nodes, H, outC] x [H, outC] => [nodes, H]

        #  Start propagating messages (runs message and aggregate)
        out = self.propagate(edge_index, x=(x_source, x_target), alpha=(alpha_l, alpha_r),size=size) # DIM: [nodes, H, outC]
        out = out.view(-1, self.heads * self.out_channels) # DIM: [nodes, H * outC]

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        # Calculate attention for edge pairs
        attention = F.leaky_relu((alpha_j + alpha_i), self.negative_slope) # EQ(1) DIM: [Edges, H]
        attention = softmax(attention, index, ptr, size_i) # EQ(2) DIM: [Edges, H] | This softmax only calculates it over all neighbourhood nodes
        attention = F.dropout(attention, p=self.dropout, training=self.training) # DIM: [Edges, H]

        # Multiple attention with node features for all edges
        out = x_j * attention.unsqueeze(-1)  # EQ(3.1) [Edges, H, outC] x [Edges, H] = [Edges, H, outC];

        return out

    def aggregate(self, inputs, index, dim_size = None):
        # EQ(3.2) For each node, aggregate messages for all neighbourhood nodes 
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, 
                                    dim_size=dim_size, reduce='sum') # inputs (from message) DIM: [Edges, H, outC] => DIM: [Nodes, H, outC]
  
        return out
```

Now with the layer all setup, in practice we will then use these convolution layers to create a neural network for us to use. Each layer consists of running the convolution layer, followed by a Relu nonlinear function and dropout. They can be stacked multiple times, and at the end we can add some output layers.

```python
class GATmodif(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim,args):
        super(GATmodif, self).__init__()
        #use our gat message passing
        ## CONV layers - replace to try different GAT versions-----------------
        self.conv1 = myGAT(input_dim, hidden_dim)
        self.conv2 = myGAT(args['heads'] * hidden_dim, hidden_dim)
        # --------------------------------------------------------------------
        # Eg. for prebuilt GAT use 
        self.conv1 = GATConv(input_dim, hidden_dim, heads=args['heads'])
        self.conv2 = GATConv(args['heads'] * hidden_dim, hidden_dim, heads=args['heads'])
        ## --------------------------------------------------------------------

        self.post_mp = nn.Sequential(
            nn.Linear(args['heads'] * hidden_dim, hidden_dim), nn.Dropout(args['dropout'] ), 
            nn.Linear(hidden_dim, output_dim))
        
    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), p=0.5, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.dropout(F.relu(x), p=0.5, training=self.training)

        # MLP output
        x = self.post_mp(x)
        return F.sigmoid(x)
        
```


Next, the metric manager helps to calculate all the required metrics at each epoch. The metric manager also has a built-in function to help calculate the best results for the entire run.


```python
class MetricManager(object):
  def __init__(self, modes=["train", "val"]):

    self.output = {}

    for mode in modes:
      self.output[mode] = {}
      self.output[mode]["accuracy"] = []
      self.output[mode]["f1micro"] = []
      self.output[mode]["f1macro"] = []
      self.output[mode]["aucroc"] = []
      #new
      self.output[mode]["precision"] = []
      self.output[mode]["recall"] = []
      self.output[mode]["cm"] = []

  def store_metrics(self, mode, pred_scores, target_labels, threshold=0.5):

    # calculate metrics
    pred_labels = pred_scores > threshold
    accuracy = accuracy_score(target_labels, pred_labels)
    f1micro = f1_score(target_labels, pred_labels,average='micro')
    f1macro = f1_score(target_labels, pred_labels,average='macro')
    aucroc = roc_auc_score(target_labels, pred_scores)
    #new
    recall = recall_score(target_labels, pred_labels)
    precision = precision_score(target_labels, pred_labels)
    cm = confusion_matrix(target_labels, pred_labels)

    # Collect results
    self.output[mode]["accuracy"].append(accuracy)
    self.output[mode]["f1micro"].append(f1micro)
    self.output[mode]["f1macro"].append(f1macro)
    self.output[mode]["aucroc"].append(aucroc)
    #new
    self.output[mode]["recall"].append(recall)
    self.output[mode]["precision"].append(precision)
    self.output[mode]["cm"].append(cm)
    
    return accuracy, f1micro,f1macro, aucroc,recall,precision,cm
  
  # Get best results
  def get_best(self, metric, mode="val"):

    # Get best results index
    best_results = {}
    i = np.array(self.output[mode][metric]).argmax()

    # Output
    for m in self.output[mode].keys():
      best_results[m] = self.output[mode][m][i]
    
    return best_results
```

After configuring the optimizer and criterion, training can run smoothly. Adam optimizer is used as it generally has a good balance between speed of convergence and stability.

```python
# Setup args and model
args={"epochs":10, 'lr':0.01, 'weight_decay':1e-5, 'prebuild':False, 'heads':2, 'num_layers': 2, 'hidden_dim': 128, 'dropout': 0.5 }
model = GATmodif(data_train.num_node_features, args['hidden_dim'], 1, args) # Change model as required, but arguments are consistent

# Push data to GPU
data_train = data_train.to(device)

# Setup training settings
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
criterion = torch.nn.BCELoss()

# Train
gnn_trainer_gatmodif = GnnTrainer(model)
gnn_trainer_gatmodif.train(data_train, optimizer, criterion, scheduler, args)
gnn_trainer_gatmodif.save_metrics("GATmodifhead2_newmetrics.results", path=FOLDERNAME + "/save_results/")
gnn_trainer_gatmodif.save_model("GATmodifhead2_newmetrics.pth", path=FOLDERNAME + "/save_results/")

Output looks like:
# epoch: 0 - loss: 0.7031 - accuracy train: 0.4681 -accuracy valid: 0.4591  - val roc: 0.5025  - val f1micro: 0.4591
# epoch: 5 - loss: 0.4539 - accuracy train: 0.9021 -accuracy valid: 0.9041  - val roc: 0.5887  - val f1micro: 0.9041
# epoch: 10 - loss: 0.3791 - accuracy train: 0.9021 -accuracy valid: 0.9041  - val roc: 0.6060  - val f1micro: 0.9041
# epoch: 15 - loss: 0.3451 - accuracy train: 0.9017 -accuracy valid: 0.9035  - val roc: 0.6035  - val f1micro: 0.9035
# epoch: 20 - loss: 0.3210 - accuracy train: 0.9021 -accuracy valid: 0.9041  - val roc: 0.6466  - val f1micro: 0.9041
```


# Attention Mechanism with GNN

Whereas GATv2 can compute a dynamic attention which overcomes the previous limitation of the GAT model, every query has a different ranking of attention coefficients of the keys. We can finally define static attention and dynamic attention which make GAT and GATv2 different.

Attention is a mechanism for computing a distribution over a set of input key vectors, given an additional query vector.

Every function f ∈ F (family of scoring functions) has a key that is always selected, regardless of the query and this cannot model situations where different keys have different relevance to different queries. Thus, to prevent this limitation, we have the dynamic attention.

We compute a dynamic attention for any set of node representations, such that there exists a constant mappings φ that map all inputs to the same output.

The GATv2 model performs better than the first version GAT, because it uses a dynamic graph attention variant that has a universal approximator attention function, it is more expressive than the other model, based on a static attention. We can see those differences in the following Figure:

<div class="figure" style="text-align: center">
<img src="img/attention.webp" alt="The deploy contexts section after clicking the Edit settings button. This section shows three settings that can be edited. The first is the production branch which is set to 'main' in a free text box. The second is deploy previews which is a radio button set to 'any pull request against your production branch/branch deploy branches (as opposed to 'none'). The third is branch deploys which is a radio button set to 'all' (as opposed to 'none' and 'let me add individual branches'). There are two buttons at the bottom of this section, Save and Cancel." width="75%" />
<p class="caption">Figure 2: Comparison of static vs dynamic attention from original paper GAT </p>
</div>



The implementation below shows how we can add the attention mechanism with Pytorch geometric

```pyhton
class myGATv2(MessagePassing):
    def __init__(self, in_channels, out_channels, heads = 1,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(myGATv2, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None
        self._alpha = None
        # self.lin_l is the linear transformation that you apply to embeddings 
        # BEFORE message passing.
        self.lin_l =  Linear(in_channels, heads*out_channels)
        self.lin_r = self.lin_l

        self.att = Parameter(torch.Tensor(1, heads, out_channels))
        self.reset_parameters()

    #initialize parameters with xavier uniform
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, size = None):
        
        H, C = self.heads, self.out_channels # DIM：H, outC
        #Linearly transform node feature matrix.
        x_source = self.lin_l(x).view(-1,H,C) # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC]
        x_target = self.lin_r(x).view(-1,H,C) # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC]
        
        #  Start propagating messages (runs message and aggregate)
        out= self.propagate(edge_index, x=(x_source,x_target),size=size) # DIM: [nodes, H, outC]
        out= out.view(-1, self.heads * self.out_channels)       # DIM: [nodes, H * outC]
        alpha = self._alpha
        self._alpha = None
        return out

    #Process a message passing
    def message(self, x_j,x_i,  index, ptr, size_i):
        #computation using previous equationss
        x = x_i + x_j                               
        x  = F.leaky_relu(x, self.negative_slope)   # See Equation above: Apply the non-linearty function
        alpha = (x * self.att).sum(dim=-1)          # Apply attnention "a" layer after the non-linearity 
        alpha = softmax(alpha, index, ptr, size_i)  # This softmax only calculates it over all neighbourhood nodes
        self._alpha = alpha
        alpha= F.dropout(alpha,p=self.dropout,training=self.training)
        # Multiple attention with node features for all edges
        out= x_j*alpha.unsqueeze(-1)  

        return out
    #Aggregation of messages
    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, 
                                    dim_size=dim_size, reduce='sum')  
        return out
        
```

# Performance

We ran various combinations of experiments, and the results are shown below. We set out 15% for a validation set from the known classified nodes. In the training process, we ran it for 100 epochs.  

Based on the model performance, we can see that GATv2 prebuilt is the best model, achieving good performance across all metrics. Due to highly imbalanced data, the f1 metrics is a very good representation of overall performance, and 0.92 on F1 macro is very good. It outperforms the GCN benchmark.

<div class="figure" style="text-align: center">
<img src="img/performance.webp" alt="The deploy contexts section after clicking the Edit settings button. This section shows three settings that can be edited. The first is the production branch which is set to 'main' in a free text box. The second is deploy previews which is a radio button set to 'any pull request against your production branch/branch deploy branches (as opposed to 'none'). The third is branch deploys which is a radio button set to 'all' (as opposed to 'none' and 'let me add individual branches'). There are two buttons at the bottom of this section, Save and Cancel." width="75%" />
<p class="caption">Figure 3: Visualization of models performances (Custom — Custom GAT layers, Prebuilt — Prebuilt GATConv layers, 3layers — Custom with 3 GNN layers (instead of 2) </p>
</div>

One observation is that the prebuilt GAT layers from PYG perform quite a bit better compared to our custom built GAT layers. This could imply some small tweaks and optimizations that they have done. It also converges much faster to its optimum performance.

There is a slight performance improvement on most metrics of GATv2 vs GAT, however for our custom built ones we were unable to demonstrate the same uplift.

Last but not least is that varying the architecture showed different results. For example, we tried a version with 3 instead of 2 GNN layers, but this seemed to lower performance. This can actually happen where more is not always better with GNN. The reason is that when we stack many layers we can experience over-smoothing depending on the size of the graph. This is caused by a big overlap in the neighborhood nodes (called the receptive field) for each target node.

# Visualization
After setting up the model, we can visualize how this looks. Firstly, we pick a time period to reduce the size of the graph as the dataset is segmented into 49 different time periods. After that we will use the NetworkX library to create a graph object, which we will use to plot the diagram.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Load model 
m1 = GATv2(data_train.num_node_features, args['hidden_dim'], 1, args).to(device).double()
m1.load_state_dict(torch.load(FOLDERNAME + "/save_results/" + "GATv2_vSK2.pth"))
gnn_t2 = GnnTrainer(m1)
output = gnn_t2.predict(data=data_train, unclassified_only=False)
output

# Get index for one time period
time_period = 28
sub_node_list = df_merge.index[df_merge.loc[:, 1] == time_period].tolist()

# Fetch list of edges for that time period
edge_tuples = []
for row in data_train.edge_index.view(-1, 2).numpy():
  if (row[0] in sub_node_list) | (row[1] in sub_node_list):
    edge_tuples.append(tuple(row))
len(edge_tuples)

# Fetch predicted results for that time period
node_color = []
for node_id in sub_node_list:
  if node_id in classified_illicit_idx: # 
     label = "red" # fraud
  elif node_id in classified_licit_idx:
     label = "green" # not fraud
  else:
    if output['pred_labels'][node_id]:
      label = "orange" # Predicted fraud
    else:
      label = "blue" # Not fraud predicted 
  
  node_color.append(label)

# Setup networkx graph
G = nx.Graph()
G.add_edges_from(edge_tuples)

# Plot the graph
plt.figure(3,figsize=(16,16)) 
nx.draw_networkx(G, nodelist=sub_node_list, node_color=node_color, node_size=6, with_labels=False)
```


The diagram below has the following legend: Green = Not illicit (not fraud), Red = illicit (Fraud), Blue = Predicted not illicit, Orange = Predicted illicit.


<div class="figure" style="text-align: center">
<img src="img/graph.webp" alt="The deploy contexts section after clicking the Edit settings button. This section shows three settings that can be edited. The first is the production branch which is set to 'main' in a free text box. The second is deploy previews which is a radio button set to 'any pull request against your production branch/branch deploy branches (as opposed to 'none'). The third is branch deploys which is a radio button set to 'all' (as opposed to 'none' and 'let me add individual branches'). There are two buttons at the bottom of this section, Save and Cancel." width="75%" />
<p class="caption">Figure 4: Visualization of original nodes and predicted nodes for time period 28. </p>
</div>

Taking a look at the graph, a majority of transactions nodes are heavily linked in a cluster. The actual fraud and the predicted fraud from the new model are fairly distributed among the central cluster and the shorter transaction chains.



# Conclusion
Graph Attention Network assign different importance to nodes of a same neighborhood, enabling a leap in model capacity and works on the entire neighboring nodes.

We show in this tutorial a detailed implementation of GAT and the improved GATv2, a more expressive one which uses dynamic attention by modifying the order of operations; it is more robust to noise edges.

The prebuilt GAT models both perform very well on the fraud dataset, achieving 0.92 F1 macro and 0.97 accuracy, beating existing benchmarks using GCN. As GATv2 outperforms GAT on several benchmarks, the GATv2 model should be considered as a baseline according to the authors, replacing the original GAT model.


# References

1- [Elliptic website](www.elliptic.co)

2- Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics, arXiv:1908.02591, 2019.

3- M. Weber G. Domeniconi J. Chen D. K. I. Weidele C. Bellei, T. Robinson, C. E. Leiserson, https://www.kaggle.com/ellipticco/elliptic-data-set, 2019

4- Graph Attention Networks, Petar Veličković and Guillem Cucurull and Arantxa Casanova and Adriana Romero and Pietro Liò and Yoshua Bengio, arXiv:1710.10903, 2018

5- Attention Is All You Need, Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin, arXiv:1706.03762, 2017.

6- How Attentive are Graph Attention Networks?, Shaked Brody, Uri Alon, Eran Yahav, arXiv:2105.14491, 2021
