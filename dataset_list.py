from torch_geometric.data import Data, Dataset, DataLoader
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
import torch


class GraphData_fromMNIST(Data):

    def __init__(self, image, label):
        super().__init__()

        adj_mat = self.compute_adjacency_mat(image)
        norm_adj_mat = self.norm_adjacency(adj_mat)
        graph = nx.from_numpy_array(norm_adj_mat, create_using=nx.DiGraph)

        # feature of node shape (N, 1)
        nodes_feature = self.create_nodes_feat_from_image(image)
        self.nodes_feature = torch.tensor(nodes_feature).unsqueeze(1)
        # tensor of size (2, num_edges)
        self.edges_index = torch.tensor(list(graph.edges)).permute(1,0)
        # label int
        self.graph_label = label

    def compute_adjacency_mat(self, image):
        """
            image: numpy array
        """
        col, row = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
        coord = np.stack((col, row), axis=2).reshape(-1, 2)
        dist = cdist(coord, coord)
        adj = ( dist <= np.sqrt(2) ).astype(float)
        return adj

    def norm_adjacency(self, adj):
        """
            adj: numpy array
        """
        deg = np.diag(np.sum(adj, axis=0))
        deg_inv_1_2 = np.linalg.inv(deg) ** (1/2)
        return deg_inv_1_2 @ adj @ deg_inv_1_2
    
    def create_nodes_feat_from_image(image):
        """
            image: numpy array
        """
        flattened_image = image.reshape(image.shape[0] * image.shape[1])
        list_flattened_image = list(flattened_image)
        return list_flattened_image

    def fill_graph_nodes_feat_list(graph, nodes_feat_list):
        """ 
            graph: networkx graph type
            nodes_feat_list: list
        """
        for i, node_feat in enumerate(nodes_feat_list):
            graph.nodes[i]['pix_int'] = node_feat


class GraphDataset(Dataset):

    def __init__(self, graph_collection):
        super().__init__()

        nodes_feat_list = []
        edges_index_list = []
        graph_label_list = []
        
        for graph in graph_collection:
            nodes_feat_list.append(graph.nodes_feature)
            edges_index_list.append(graph.edges_index)
            graph_label_list.append(graph.label)

        self.nodes_feature_list = nodes_feat_list
        self.edges_index_list = edges_index_list

    def __len__(self):
        return len(self.nodes_feature_list)
    
    def __getitem__(self, index):
        return self.nodes_feature_list[index], self.edges_index_list[index]

class GraphDataLoader(DataLoader):
    
    def __init__(self, graph_collection, batch_size=1, shuffle=False):
        graph_dataset = GraphDataset(graph_collection)

        super().__init__(graph_dataset, batch_size, shuffle)
