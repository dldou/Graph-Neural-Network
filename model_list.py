import torch.nn as nn
import torch


class GCN_model(nn.Module):

    def __init__(self, num_layer, num_of_nodes, num_ofclass):
        super(GCN_model, self).__init__()
        self.fc = nn.Linear(in_features=num_of_nodes, out_features=num_ofclass)
    
    def forward(self, nodes_feat_list, edges_index_list, graph_label_list):
        # D ** (-1 / 2) of shape (num_of_nodes) only for one matrix as all graph got the same structure
        graph_degrees = self.compute_degree(edges_index_list, num_of_nodes=nodes_feat_list.shape[1])

        # expand degree matrix to perform Hadamard product
        graph_degrees = graph_degrees.unsqueeze(-1)
        # Hadamard product h_l * (D ** (-1 / 2))
        nodes_feat_list = nodes_feat_list * graph_degrees

        # aggregation 
        nodes_features_aggregated = self.aggregate_neighbors(nodes_feat_list, edges_index_list[0,:,:])

        # Second Hadamard product
        nodes_features_aggregated = nodes_features_aggregated * graph_degrees

        # message passing
        nodes_features_aggregated = nodes_features_aggregated.permute(0,2,1)
        nodes_features_output = self.fc(nodes_features_aggregated)

        return nodes_features_output
    
    def aggregate_neighbors(self, nodes_feature_tens, edges_index):
        """ 
            nodes_feature_tens: tensor of shape (batch_size, num_nodes, num_features=1)
            edges_index: tensor of shape (2, num_edges)
        """
        num_of_nodes = nodes_feature_tens.shape[1]
        squeeze_flag = False
        
        if nodes_feature_tens.shape[-1] == 1:
            nodes_feature_tens = nodes_feature_tens.squeeze(-1)
            squeeze_flag = True
        
        # expand nodes_feature_tens to be able to use scatter_add_ function
        # (batch_size, num_of_nodes) --> (batch_size, num_of_edges)
        expanded_nodes_feature_tens = nodes_feature_tens[:,edges_index[1,:]]
        
        # nodes on which to aggregate
        target_index = edges_index[0,:].unsqueeze(0)
        
        # shape (batch_size, num_of_edges)
        nodes_feature_tens_output = torch.zeros_like(expanded_nodes_feature_tens, dtype=nodes_feature_tens.dtype)
        
        # for all node i, sum all j from i's neighborhood (in place)
        nodes_feature_tens_output.scatter_add_(dim=1, index=target_index, src=expanded_nodes_feature_tens)
        # crop from shape (batch_size, num_of_edges) to (batch_size, num_of_nodes)
        nodes_feature_tens_output = nodes_feature_tens_output[:, 0:num_of_nodes]
        
        if squeeze_flag:
            nodes_feature_tens_output = nodes_feature_tens_output.unsqueeze(-1)

        return nodes_feature_tens_output

    def compute_degree(self, edges_index, num_of_nodes):
        """
            Compute the degree tensor 
            /!\ doesn't fit unconnected graph
            edges_index: tensor of shape (batch_size, 2, number of edges) 
        """
        _, degree = torch.unique(edges_index[0,0,:], return_counts=True)
        assert degree.shape[0] == num_of_nodes, f'Expected degree matrix with shape=({num_of_nodes}) got {degree.shape}.'
        return degree ** (-1 / 2)

