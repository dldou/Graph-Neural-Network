from torch_geometric.data import Data, Dataset, DataLoader
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
import torch
import pickle as pkl

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
    
    def create_nodes_feat_from_image(self, image):
        """
            image: numpy array
        """
        flattened_image = image.reshape(image.shape[0] * image.shape[1])
        list_flattened_image = list(flattened_image)
        return list_flattened_image

    def fill_graph_nodes_feat_list(self, graph, nodes_feat_list):
        """ 
            graph: networkx graph type
            nodes_feat_list: list
        """
        for i, node_feat in enumerate(nodes_feat_list):
            graph.nodes[i]['pix_int'] = node_feat


# graph_collection = []
# num_graphs = 0
# num_samples = len(train_set)
# for image, label in train_set:
#     graph_collection.append(GraphData_fromMNIST(image.squeeze(), label))
#     num_graphs += 1
#     print("\r" + "{:.1f} %".format( (num_graphs / num_samples) * 100), end="")
#     #print("\r" + "{} / {}".format(num_graphs, len(train_set)), end="") 

# with open('graph_collection_dataset.pkl', 'wb') as f:
#     pkl.dump(graph_collection, f)


class GraphDataset(Dataset):

    def __init__(self, graph_collection):
        #super(GraphDataset, self).__init__()
        nodes_feat_list = []
        edges_index_list = []
        graph_label_list = []
        
        for graph in graph_collection:
            nodes_feat_list.append(graph.nodes_feature)
            edges_index_list.append(graph.edges_index)
            graph_label_list.append(graph.graph_label)

        self.nodes_feature_list = nodes_feat_list
        self.edges_index_list = edges_index_list
        self.graph_label_list = graph_label_list

    def __len__(self):
        return len(self.nodes_feature_list)
    
    def __getitem__(self, index):
        return self.nodes_feature_list[index], self.edges_index_list[index], self.graph_label_list[index]

    # In PyG Dataset definition len and get are abstract methods (?)
    def len(self):
        return self.__len__()
    
    def get(self, idx):
        return self.__getitem__(idx)


class GraphDataLoader(DataLoader):
    
    def __init__(self, graph_collection, batch_size=1, shuffle=False):
        graph_dataset = GraphDataset(graph_collection)

        super().__init__(graph_dataset, batch_size, shuffle)



def compute_graph_collection_and_save_pickle_file(dataset, filename, ratio=1):
    graph_collection = []
    num_seen_graphs = 0
    num_samples = len(dataset)

    for image, label in dataset:
        if (num_seen_graphs / num_samples) < ratio:
            graph_collection.append(GraphData_fromMNIST(image.squeeze(), label))
            num_seen_graphs += 1
            print("\r" + "{:.2f} %".format( (num_seen_graphs / num_samples) * 100), end="")
        else:
            break

    with open(filename, 'wb') as f:
        pkl.dump(graph_collection, f)


def load_graph_collection_from_pickle_file(filename):
    with open(filename, 'rb') as f:
        graph_collection = pkl.load(f)    
    return graph_collection