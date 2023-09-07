import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
import os
from config import CONFIG, MODEL_PARAMS
import helper

# 'encoder layers'
# fed_layer1 = GCNConv(MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["Modality1"]["n_features"])
# fed_layer2 = GCNConv(MODEL_PARAMS["intermediate_resolution"], MODEL_PARAMS["Modality1"]["n_features"])
# encoder2_layer3 = GCNConv(MODEL_PARAMS["Modality2"]["n_features"], MODEL_PARAMS["intermediate_resolution"])
# encoder3_layer3 = GCNConv(MODEL_PARAMS["Modality3"]["n_features"], MODEL_PARAMS["intermediate_resolution"])

# 'decoder layers'
# decoder_layer1 = GCNConv(MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["Modality1"]["n_features"])
# decoder_layer2 = GCNConv(MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["intermediate_resolution"])
# decoder2_layer3 = GCNConv(MODEL_PARAMS["intermediate_resolution"], MODEL_PARAMS["Modality2"]["n_features"])
# decoder3_layer3 = GCNConv(MODEL_PARAMS["intermediate_resolution"], MODEL_PARAMS["Modality3"]["n_features"])

dropout_prob = MODEL_PARAMS["dropout_prob"]
cbt_resolution = MODEL_PARAMS["CBT_resolution"]
roi1 = MODEL_PARAMS["Modality1"]["N_ROIs"]
roi2 = MODEL_PARAMS["Modality2"]["N_ROIs"]
roi3 = MODEL_PARAMS["Modality3"]["N_ROIs"]
batch_size = CONFIG["batch_size"]

m = nn.LeakyReLU(0.05, inplace=True)



def repeat_to_batch_size(x, batch_size):
    if x.shape[0] > batch_size:
        raise ValueError(f"Input tensor has more than {batch_size} rows!")
    repeats = (batch_size + x.shape[0] - 1) // x.shape[0]
    repeated_x = torch.repeat_interleave(x, repeats=repeats, dim=0)[:batch_size, :]
    return repeated_x


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        "encoder"
        self.conv1 = GCNConv(MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["Modality1"]["n_features"])
        "decoder"
        self.conv12 = GCNConv(MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["Modality1"]["n_features"])
        "normalizer"
        self.linear1 = nn.Linear(batch_size* MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["intermediate_channel"])
        self.linear2 = nn.Linear( MODEL_PARAMS["intermediate_channel"], MODEL_PARAMS["Modality1"]["n_features"])

    def encode(self, x, edge_index):

        x1 = torch.sigmoid(self.conv1(x, edge_index))
        z = F.dropout(x1, p=dropout_prob, training=self.training)
        
        return z
    
    def bacth_based_normalize(self, z):
        if z.shape[0] < batch_size:
            z = repeat_to_batch_size(z, batch_size)
        z = z.view(-1)
        i1 = self.linear1(z)
        i1 = torch.sigmoid(i1)

        i2 = self.linear2(i1)
        i2 = torch.sigmoid(i2)

        return i1, i2.view(1, -1)
    
    # def decode(self, z, edge_index):
        
    #     y1 = torch.sigmoid(self.conv12(z, edge_index))
    #     y1 = F.dropout(y1, p=dropout_prob, training=self.training)

    #     return y1

    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features with shape [1, m*(m-1)/2] (we treat the feature vector as node featues of an 1-node graph, m is resolution)
            edge_index: Graph connectivities with shape [2, number_of_edges](each only connects to itself)
        """

        z = self.encode(x, edge_index)
        i1, cbt = self.bacth_based_normalize(z)
 

        # Antivectorze
        x_matrix = helper.batch_antiVectorize(x, roi1)
        z_matrix = helper.batch_antiVectorize(z, cbt_resolution)
        cbt_matrix = helper.antiVectorize(cbt, cbt_resolution)

        return x_matrix, z_matrix, z_matrix, i1, cbt_matrix




class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        "encoder"
        self.conv23 = GCNConv(MODEL_PARAMS["Modality2"]["n_features"], MODEL_PARAMS["intermediate_resolution"])
        self.conv2 = GCNConv(MODEL_PARAMS["intermediate_resolution"], MODEL_PARAMS["Modality1"]["n_features"])
        self.conv1 = GCNConv(MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["Modality1"]["n_features"])
        "decoder"
        self.conv24 = GCNConv(MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["Modality1"]["n_features"])
        self.conv25 = GCNConv(MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["intermediate_resolution"])
        self.conv26 = GCNConv(MODEL_PARAMS["intermediate_resolution"], MODEL_PARAMS["Modality2"]["n_features"])
        "normalizer"
        self.linear1 = nn.Linear(batch_size* MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["intermediate_channel"])
        self.linear2 = nn.Linear( MODEL_PARAMS["intermediate_channel"], MODEL_PARAMS["Modality1"]["n_features"])

    
    def encode(self, x, edge_index):

        x1 = torch.sigmoid(self.conv23(x, edge_index))
        x1 = F.dropout(x1, p=dropout_prob, training=self.training) 

        x2 = torch.sigmoid(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=dropout_prob,training=self.training)

        x3 = torch.sigmoid(self.conv1(x2, edge_index))
        z = F.dropout(x3, p=dropout_prob,training=self.training)

        return z
    
    def bacth_based_normalize(self, z):
        if z.shape[0] < batch_size:
            z = repeat_to_batch_size(z, batch_size)
        z = z.view(-1)
        i1 = self.linear1(z)
        i1 = torch.sigmoid(i1)

        i2 = self.linear2(i1)
        i2 = torch.sigmoid(i2)

        return i1, i2.view(1, -1)

    def decode(self, z, edge_index):

        y1 = torch.sigmoid(self.conv24(z, edge_index))
        y1 = F.dropout(y1, p=dropout_prob, training=self.training)

        y2 = torch.sigmoid(self.conv25(y1, edge_index))
        y2 = F.dropout(y2, p=dropout_prob, training=self.training)

        y3 = torch.sigmoid(self.conv26(y2, edge_index))
        y3 = F.dropout(y3, p=dropout_prob, training=self.training)

        return y3

    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features with shape [1, m*(m-1)/2] (we treat the feature vector as node featues of an 1-node graph, m is resolution)
            edge_index: Graph connectivities with shape [2, number_of_edges](each only connects to itself)
        """

        z = self.encode(x, edge_index)
        i1, cbt = self.bacth_based_normalize(z)
        y = self.decode(z, edge_index)

        # Antivectorze
        x_matrix = helper.batch_antiVectorize(x, roi2)
        z_matrix = helper.batch_antiVectorize(z, cbt_resolution)
        y_matrix = helper.batch_antiVectorize(y, roi2)
        cbt_matrix = helper.antiVectorize(cbt, cbt_resolution)

        return x_matrix, z_matrix, y_matrix, i1, cbt_matrix


class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        "encoder"
        self.conv33 = GCNConv(MODEL_PARAMS["Modality3"]["n_features"], MODEL_PARAMS["intermediate_resolution"])
        self.conv2 = GCNConv(MODEL_PARAMS["intermediate_resolution"], MODEL_PARAMS["Modality1"]["n_features"])
        self.conv1 = GCNConv(MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["Modality1"]["n_features"])
        "decoder"
        self.conv34 = GCNConv(MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["Modality1"]["n_features"])
        self.conv35 = GCNConv(MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["intermediate_resolution"])
        self.conv36 = GCNConv(MODEL_PARAMS["intermediate_resolution"], MODEL_PARAMS["Modality3"]["n_features"])
        "normalizer"
        self.linear1 = nn.Linear(batch_size* MODEL_PARAMS["Modality1"]["n_features"], MODEL_PARAMS["intermediate_channel"])
        self.linear2 = nn.Linear( MODEL_PARAMS["intermediate_channel"], MODEL_PARAMS["Modality1"]["n_features"])
    
    def encode(self, x, edge_index):

        x1 = torch.sigmoid(self.conv33(x, edge_index))
        x1 = F.dropout(x1, p=dropout_prob, training=self.training) 

        x2 = torch.sigmoid(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=dropout_prob,training=self.training)

        x3 = torch.sigmoid(self.conv1(x2, edge_index))
        z = F.dropout(x3, p=dropout_prob,training=self.training)

        return z
    
    def bacth_based_normalize(self, z):
        if z.shape[0] < batch_size:
            z = repeat_to_batch_size(z, batch_size)
        z = z.view(-1)
        i1 = self.linear1(z)
        i1 = torch.sigmoid(i1)

        i2 = self.linear2(i1)
        i2 = torch.sigmoid(i2)

        return i1, i2.view(1, -1)

    def decode(self, z, edge_index):

        y1 = m(self.conv34(z, edge_index))
        y1 = F.dropout(y1, p=dropout_prob, training=self.training)

        y2 = m(self.conv35(y1, edge_index))
        y2 = F.dropout(y2, p=dropout_prob, training=self.training)

        y3 = m(self.conv36(y2, edge_index))
        y3 = F.dropout(y3, p=dropout_prob, training=self.training)

        return y3

    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features with shape [1, m*(m-1)/2] (we treat the feature vector as node featues of an 1-node graph, m is resolution)
            edge_index: Graph connectivities with shape [2, number_of_edges](each only connects to itself)
        """

        z = self.encode(x, edge_index)
        i1, cbt = self.bacth_based_normalize(z)
        y = self.decode(z, edge_index)

        # Antivectorze
        x_matrix = helper.batch_antiVectorize(x, roi3)
        z_matrix = helper.batch_antiVectorize(z, cbt_resolution)
        y_matrix = helper.batch_antiVectorize(y, roi3)
        cbt_matrix = helper.antiVectorize(cbt, cbt_resolution)

        return x_matrix, z_matrix, y_matrix, i1, cbt_matrix
        