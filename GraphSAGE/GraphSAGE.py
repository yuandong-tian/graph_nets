import torch
import torch.nn as nn
from torch.nn import init

import numpy as np
import os
import time
import json
import random
import torch.nn.functional as F
from sklearn.metrics import f1_score
from collections import defaultdict

#from graphsage.encoders import Encoder
#from graphsage.aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
      #  print ("\n unl's size=",len(unique_nodes_list))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], 
                self.num_sample)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("./cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = [float(x) for x in info[1:-1]]
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("./cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    return feat_data, labels, adj_lists, train, val, test

def load_data(prefix, label_idx=None):
    root = "/checkpoint/yuandong/datasets/" + prefix
    graph = json.load(open(os.path.join(root, f"{prefix}-G.json")))
    id_map = json.load(open(os.path.join(root, f"{prefix}-id_map.json")))
    class_map = json.load(open(os.path.join(root, f"{prefix}-class_map.json")))
    feat_data = np.load(os.path.join(root, f"{prefix}-feats.npy"))

    num_nodes = feat_data.shape[0]
    # For all nodes that are not used, assign them -1.
    # They won't get sampled since train/test/eval won't have it. 
    labels = -np.ones((num_nodes,1), dtype=np.int64)

    # Edge connections
    adj_lists = defaultdict(set)
    for link in graph["links"]:
        s_idx = link["source"]
        t_idx = link["target"]
        adj_lists[s_idx].add(t_idx)
        adj_lists[t_idx].add(s_idx)

    train = []
    test = []
    val = []
    isolated = []

    for node in graph["nodes"]:
        id = str(node["id"]) 
        idx = id_map[id]

        # do not include the idx to any sets, if it is not connected with anyone else. 
        if len(adj_lists[idx]) == 0:
            isolated.append(idx)
            continue

        if node["test"]:
            test.append(idx)
        elif node["val"]:
            val.append(idx)
        else:
            train.append(idx)
        if label_idx is None:
            labels[idx] = class_map[id]
        else:
            # ppi has a lot of labels at each vertex
            labels[idx] = class_map[id][label_idx] # [0]

    return feat_data, labels, adj_lists, train, val, test, isolated

def run_cora():
    cuda = True
    np.random.seed(1)
    random.seed(1)
    # feat_data, labels, adj_lists, train, val, test = load_cora()
    prefix = "reddit"
    load = True
    filename = prefix + ".th"

    if not load:
        all_data = load_data(prefix, None)
        print(f"Saving {filename}")
        torch.save(all_data, filename)
    else:
        print(f"Loading {filename}")
        all_data = torch.load(filename)

    feat_data, labels, adj_lists, train, val, test, isolated = all_data 

    num_nodes, d = feat_data.shape
    num_classes = max(labels).item() + 1
    hidd_dim = 128

    print(f"#node: {num_nodes}, #isolated: {len(isolated)}, #num_class: {num_classes}, feature_dim: {d}, hidd_dim: {hidd_dim}")

    features = nn.Embedding(num_nodes, d)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    if cuda:
        features.cuda()

    agg1 = MeanAggregator(features, cuda=cuda)
    enc1 = Encoder(features, d, hidd_dim, adj_lists, agg1, gcn=True, cuda=cuda)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=cuda)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, hidd_dim, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=cuda)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(num_classes, enc2)
    graphsage.cuda()

    params = filter(lambda p : p.requires_grad, graphsage.parameters())
    # optimizer = torch.optim.SGD(params, lr=0.7)
    optimizer = torch.optim.Adam(params, lr=0.01)
    times = []
    for batch in range(10):
        batch_nodes = train[:256]
        # print(batch_nodes)
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        x = torch.LongTensor(labels[np.array(batch_nodes)])
        if cuda:
            x = x.cuda()

        if (x < 0).sum().item() != 0:
            print("Something wrong!")
            import pdb
            pdb.set_trace()

        loss = graphsage.loss(batch_nodes, x)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print (batch, loss.item())

    graphsage = graphsage.to('cpu')
    val_output = graphsage.forward(val) 
    print ("Validation F1:", f1_score(labels[val], val_output.cpu().numpy().argmax(axis=1), average="micro"))
    print ("Average batch time:", np.mean(times))

if __name__ == "__main__":
    run_cora()
