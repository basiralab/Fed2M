import uuid
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import OrderedDict
from sklearn.manifold import TSNE
import pandas as pd
from config import MODEL_PARAMS
import pickle
import helper
from config import CONFIG
from model import Model1, Model2, Model3
import random
# Setting up working environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#These two options shoul be seed to ensure reproducible (If you are using cudnn backend)
#https://pytorch.org/docs/stable/notes/randomness.html
#We used 35888 as the seed when we conducted experiments
np.random.seed(35813)
torch.manual_seed(35813)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
MODEL_WEIGHT_BACKUP_PATH = "./output"
TEMP_FOLDER = "./temp"

if not os.path.exists(MODEL_WEIGHT_BACKUP_PATH):
    os.makedirs(MODEL_WEIGHT_BACKUP_PATH)

if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)


cbt_resolution = CONFIG["CBT_resolution"]
fed_rounds = CONFIG["fed_rounds"]
epochs_per_round = CONFIG["epochs_per_round"]
batch_size = CONFIG["batch_size"]
local_sample_size = CONFIG["local_sample_size"]
global_sample_size = CONFIG["global_sample_size"]
n_clusters = CONFIG["n_clusters"]
sampling_method = CONFIG["sampling_method"]
model_name = CONFIG["model_name"]
# ablated = CONFIG["ablated"]
early_stop = CONFIG["early_stop"]
freeze = CONFIG["freeze"]
n_folds = CONFIG["n_folds"]
m1 = CONFIG["client1"]["N_ROIs"]
m2 = CONFIG["client2"]["N_ROIs"]
m3 = CONFIG["client3"]["N_ROIs"]

work_path = MODEL_WEIGHT_BACKUP_PATH + "/"  + sampling_method + "/"+ model_name
if not os.path.exists(work_path):
    os.makedirs(work_path)
with open(work_path + "model_params.txt", 'w') as f:
    print(MODEL_PARAMS, file=f)
with open(work_path + "model_config.txt", 'w') as f:
    print(CONFIG, file=f)


def frobenious_distance(input_tensor, output_tensor):
    """Calculate the mean Frobenius distance between two tensors.
        used in the calculation of centerdness loss and reconstruction loss
    """
    assert input_tensor.shape == output_tensor.shape, "Input and output tensors must have the same shape"
    
    # Compute element-wise difference
    diff = input_tensor - output_tensor
    
    # Square the differences
    squared_diff = torch.pow(diff, 2)
    
    # Sum the squared differences along all dimensions
    sum_squared_diff = torch.sum(squared_diff)
    
    # Take the square root of the summed squared differences
    frobenius_distance = torch.sqrt(sum_squared_diff)

    return frobenius_distance


def calculate_local_centeredness_loss(input_tensor, output_tensor):
    """Calculate the mean Frobenius distance between two tensors.
    """
    assert input_tensor.shape == output_tensor.shape, "Input and output tensors must have the same shape"
    
    # Compute element-wise difference
    diff = input_tensor - output_tensor  # i*n*35*35
    
    # Square the differences
    squared_diff = torch.pow(diff, 2)  # i*n*35*35
    
    # Sum the squared differences along all dimensions
    sum_squared_diff = torch.sum(squared_diff, dim=(1, 2, 3))  # i
    
    # Take the square root of the summed squared differences
    frobenius_distance = torch.sqrt(sum_squared_diff)  # i

    return frobenius_distance


def calculate_reconstruction_loss(input_tensor, output_tensor):
    """Calculate the mean Frobenius distance between two tensors.
    """
    assert input_tensor.shape == output_tensor.shape, "Input and output tensors must have the same shape"
    
    # Compute element-wise difference
    diff = input_tensor - output_tensor  # i*35*35
    
    # Square the differences
    squared_diff = torch.pow(diff, 2)  # i*35*35
    
    # Sum the squared differences along all dimensions
    sum_squared_diff = torch.sum(squared_diff, dim = (1, 2))  # i
    
    # Take the square root of the summed squared differences
    frobenius_distance = torch.sqrt(sum_squared_diff) # i

    return frobenius_distance

def calculate_global_centeredness_loss(sampled_is, i1):
    # Compute pairwise L2 loss and sum them
    total_l2_loss = 0
    for i in range(len(sampled_is) - 1):
        total_l2_loss += torch.norm(sampled_is[i] - i1)
    
    return total_l2_loss

def normalized_node_strength(connectivity_map):
    """For the calculation of topology loss

    Args:
        connectivity_map (tensor): n*m*m
    """
    map_dist = connectivity_map.sum(dim=2)  # sum along the last dimension
    cbt_probs = map_dist / map_dist.sum(dim=1, keepdim=True)  # sum along the second dimension
    return cbt_probs


def generate_cbt_median(model, data_loader, edge_index):
    """
        each data comes from a modality with different resolution

        data1: Node features with shape [1, m*(m-1)/2]
        data2: Node features with shape [1, m*(m-1)/2]
        data3: Node features with shape [1, m*(m-1)/2]
        edge_index: Graph connectivities with shape [2, number_of_edges](each only connects to itself)
    """
    model.eval()
    cbts = []
    embeddings = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        _, z, _, _, cbt = model(data, edge_index)  # n*35*35
        cbts.append(cbt)
        embeddings.append(z)
    cbts = torch.stack(cbts, dim = 0)
    embeddings = torch.concatenate(embeddings, axis=0)
    final_cbt = torch.median(cbts, dim = 0)[0]
    return final_cbt , embeddings.detach()


def evaluate_local_centeredness(model, data_loader, edge_index, generated_cbt):
    """ pass graphs with 160*160 and 268*268 to Encoder_160 and Encoder_268
        compute the mean frobenious distance between cbt and encoded graphs

    Args:
        data1_casted(geometric_data): resolution 35
        data2_casted (geometric_data): resolution 160
        data3_casted (geometric_data): resolution 268
        generated_cbt (geometric_data): generated median cbt  

    Returns:
        torch.tensor: centeredness score(lower is better)
    """

    scores = []
    
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)

        z = model.encode(data, edge_index)

        for i in range(z.shape[0]):
            vec = z[i:(i+1), :]
            matrix = helper.antiVectorize(vec, cbt_resolution)
            scores.append(frobenious_distance(matrix, generated_cbt))

    return (sum(scores)/len(scores)).detach().cpu().clone().numpy()

def evaluate_reconstruction(model, data_loader, edge_index):
        """Compute the MAE between the ground-truth connetivity matrices and reconstructed
           matrices.

        Args:
            data1_casted (geometric_data): resolution 35
            data2_casted (geometric_data): resolution 160
            data3_casted (geometric_data): resolution 268

        Returns:
            torch.tensor: reconstruction score(lower is better)
        """
        scores = []

        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
    
            x, _, y, _, _ = model(data, edge_index)
            scores.append(torch.mean(torch.abs(x - y)))
        
        recon = (sum(scores)/len(scores)).detach().cpu().clone().numpy()  
     
        return recon

def evaluate_topology(model, data, m, edge_index, generated_cbt):
        """quntify the difference between ground-truth connectivity matrices and CBT-based reconstructed
        matrices

        Args:
            data2_casted (geometric_data): resolution 160
            data3_casted (geometric_data): resolution 268
            generated_cbt (geometric_data): generated median cbt  
        Returns:
            torch.tensor: topology score(lower is better)
        """

        cbt_vec = helper.vectorize(generated_cbt)
        y = model.decode(cbt_vec.clone(), edge_index)
        y_matrix = helper.batch_antiVectorize(y, m)
        p = normalized_node_strength(y_matrix)
    
        scores = []
        for i in range(len(data)):
            x = torch.tensor(data[i]).unsqueeze(dim = 0)
            x_matrix = helper.batch_antiVectorize(x, m)
            target_p = normalized_node_strength(x_matrix)
     
            scores.append(torch.abs(target_p - p).sum())
        
        return (sum(scores)/len(scores)).detach().cpu().clone().numpy()


def evaluate_global_centeredness(generated_cbt1, generated_cbt2, generated_cbt3):
    cbt_list = [generated_cbt1, generated_cbt2, generated_cbt3]
    total_l2_loss = 0
    for i in range(len(cbt_list) - 1):
        for j in range(i + 1, len(cbt_list)):
            total_l2_loss += torch.norm(cbt_list[i] - cbt_list[j])
    return total_l2_loss.item()


def frobenious_distance(input_tensor, output_tensor):
        """Calculate the mean Frobenius distance between two tensors.
            used in the calculation of centerdness loss and reconstruction loss
        """
        assert input_tensor.shape == output_tensor.shape, "Input and output tensors must have the same shape"
        
        # Compute element-wise difference
        diff = input_tensor - output_tensor
        
        # Square the differences
        squared_diff = torch.pow(diff, 2)
        
        # Sum the squared differences along all dimensions
        sum_squared_diff = torch.sum(squared_diff)
        
        # Take the square root of the summed squared differences
        frobenius_distance = torch.sqrt(sum_squared_diff)

        return frobenius_distance

'''
federated lerning
'''
class Client:
    def __init__(self, model, N_ROIs,  optimizer, config, train_data):
        '''
        model should be a class
        '''
        self.params = config
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.m = N_ROIs
    
    def update_weights(self, data, edge_index , global_sample,  sampling_method, freezed, ablated):
        data = data.to(device)
        x, z , y, i1, cbt = self.model(data, edge_index)
        # helper.show_image(y[0].detach().cpu().numpy())

        # Calculate local centredness loss

        regularized_z_matrix = []
        for i in range(z.shape[0]):
            sampled_z_matrix = []
            assert sampling_method in ['kmeans', 'hierarchical', 'random']
            if sampling_method == 'kmeans':
                sampled_subject_ids = helper.k_means_clustering_sampling(self.train_data, n_clusters, local_sample_size)
            elif sampling_method == 'hierarchical':
                sampled_subject_ids = helper.hierarchical_clustering_sampling(self.train_data, n_clusters, local_sample_size)
            elif sampling_method == 'random':
                sampled_subject_ids = helper.random_sampling(self.train_data, local_sample_size)
            
            sampled_data = torch.vstack([torch.tensor(self.train_data[i]).float().to(device) for i in sampled_subject_ids])
            sampled_z_vec = self.model.encode(sampled_data, edge_index)
            sampled_z_matrix = helper.batch_antiVectorize(sampled_z_vec, cbt_resolution)
            z_matrix = torch.cat([sampled_z_matrix, z[i:i+1, :, :]], dim = 0)
            regularized_z_matrix.append(z_matrix)
            
        regularized_z_matrix = torch.stack(regularized_z_matrix, dim = 0)

        expanded_cbt_matrix =  cbt.expand(local_sample_size+1, z.shape[0], cbt_resolution, cbt_resolution)
        expanded_cbt_matrix = expanded_cbt_matrix.permute(1, 0, 2, 3)

        local_centeredness_loss = calculate_local_centeredness_loss(regularized_z_matrix, expanded_cbt_matrix)

        # Calculate reconstruction loss
        reconstruction_loss = calculate_reconstruction_loss(x, y)

        # Calculate topology loss
        target_p = normalized_node_strength(x)
        p = normalized_node_strength(y)
        topology_loss = torch.abs(target_p - p).sum(dim = 1)

        # Calculate global loss
        if global_sample is not None:
            global_sample = [sample.to(device) for sample in global_sample]
            global_centeredness_loss = calculate_global_centeredness_loss(global_sample, i1)
        
        else:
            global_centeredness_loss = torch.tensor(0).to(device)

        # Sum up the three loss
        
        loss = reconstruction_loss+ self.params["lambda1"]*topology_loss+ self.params["lambda2"]*local_centeredness_loss 

        local_centeredness_loss = local_centeredness_loss.detach().cpu().clone().tolist()
        reconstruction_loss = reconstruction_loss.detach().cpu().clone().tolist()
        topology_loss = topology_loss.detach().cpu().clone().tolist()
        global_centeredness_loss = global_centeredness_loss.detach().cpu().clone()
        total_local_loss = loss.detach().cpu().clone().tolist()

        #Backprob
        if freezed:
            pass
        else:
            self.optimizer.zero_grad()
            if ablated:
                losses = torch.sum(loss)
            else:
                losses = torch.sum(loss) + self.params['lambda3']*global_centeredness_loss
            losses.backward()
            self.optimizer.step()

        return i1, local_centeredness_loss, reconstruction_loss, topology_loss, global_centeredness_loss, total_local_loss
            
    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, state_dict):
        updated_state_dict = {}
        for key,_ in state_dict.items():
            if key in self.model.state_dict().keys():
                updated_state_dict.update({key:_})
            else:
                pass
        self.model.load_state_dict(updated_state_dict)
    

class Server:
    def __init__(self, clients):
        self.clients = clients

    def broadcast_weights(self):
        avg_state_dict = self.average_weights()
        for client in self.clients:
            client.set_weights(avg_state_dict)

    def average_weights(self):
        """
        Average the weights of the models from all clients for the common layers
        """
        # Get the state_dict from each client's model
        state_dicts = [client.model.state_dict() for client in self.clients]

        # Get keys from all state dict
        keys = set()
        for client in state_dicts:
            keys.update(set(client.keys()))

        # Initialize a new state_dict to store the average weights
        average_state_dict = OrderedDict()

        # For each key (layer) in the state_dict
        for key in keys:
            # Get the list of state_dicts for this layer from clients that have this layer
            current_layer_state_dicts = [client[key] for client in state_dicts if key in client]

            # If no clients have this layer, skip it
            if not current_layer_state_dicts:
                continue
            
            # Average the weights from all clients for this layer
            average_state_dict[key] = torch.stack(current_layer_state_dicts).mean(dim=0)

        return average_state_dict
    
    # @staticmethod
    # def global_centeredness_loss(i1, i2, i3):
    #     i_list = [i1, i2, i3]
    #     # Compute pairwise L2 loss and sum them
    #     total_l2_loss = 0
    #     for i in range(len(i_list) - 1):
    #         for j in range(i + 1, len(i_list)):
    #             total_l2_loss += torch.norm(i_list[i] - i_list[j])
        
    #     return total_l2_loss.item()
    @staticmethod
    def distribute_representation(client_no, i1_list, i2_list, i3_list, sample_size):
        if client_no == 1:
            i_pool = i2_list+i3_list
        elif client_no == 2: 
            i_pool = i1_list+i3_list
        elif client_no == 3:
            i_pool = i1_list+i2_list

        sampled_is = random.sample(i_pool, sample_size)

        return sampled_is


        
    
        
'''
demo
'''
def train(model_id,i,  X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, is_federated, ablated):
    """_summary_

    Args:
        i (int): fold number
        X1_train (_type_): _description_
        X1_test (_type_): _description_
        X2_train (_type_): _description_
        X2_test (_type_): _description_
        X3_train (_type_): _description_
        X3_test (_type_): _description_
        federated_learning (_type_): _description_
    """
    
    if is_federated:
        print("FEDERATION ")
        if ablated:
            print("Ablated")
            save_path = work_path + "/" + "ablated_fed"
        else:
            print("Global centeredness loss")
            save_path = work_path + "/" + "global_fed"
    else:
        print("NO FEDERATION ")
        save_path = work_path + "/" + "nofed"

    cbt_path1 = save_path + "/" + "client1_cbts" + "/"
    cbt_path2 = save_path + "/" + "client2_cbts" + "/"
    cbt_path3 = save_path + "/" + "client3_cbts" + "/"
    loss_path = save_path + "/" + "loss" +"/"
    eval_path = save_path + "/" + "eval" +"/"
    model_path = save_path + "/" + "model" +"/"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(cbt_path1):
        os.makedirs(cbt_path1)
    if not os.path.exists(cbt_path2):
        os.makedirs(cbt_path2)
    if not os.path.exists(cbt_path3):
        os.makedirs(cbt_path3)
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    

    total_epochs = fed_rounds*epochs_per_round

    # create trainset and testset loader
    X1_train_loader = helper.get_loader(X1_train, batch_size)
    X1_test_loader = helper.get_loader(X1_test, batch_size)

    X2_train_loader = helper.get_loader(X2_train, batch_size)
    X2_test_loader = helper.get_loader(X2_test, batch_size)

    X3_train_loader = helper.get_loader(X3_train, batch_size)
    X3_test_loader = helper.get_loader(X3_test, batch_size)

    # get edge index( each node only connects to itself)
    edge_index = helper.get_edge_index(batch_size, device)

    model1 = Model1()
    model1.to(device)
    model2 = Model2()
    model2.to(device)
    model3 = Model3()
    model3.to(device)

    optimizer1 = torch.optim.AdamW(model1.parameters(), lr= CONFIG["client1"]["learning rate"], weight_decay= 0.00)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr= CONFIG["client2"]["learning rate"], weight_decay= 0.00)
    optimizer3 = torch.optim.AdamW(model3.parameters(), lr= CONFIG["client3"]["learning rate"], weight_decay= 0.00)

    client1 = Client(model1, m1, optimizer1, CONFIG["client1"], X1_train)
    client2 = Client(model2, m2, optimizer2, CONFIG["client2"], X2_train)
    client3 = Client(model3, m3, optimizer3, CONFIG["client3"], X3_train)
    server = Server([client1, client2, client3])

    test_errors1 = []
    test_errors2 = []
    test_errors3 = []
    freezed1 = False
    freezed2 = False
    freezed3 = False

    epoch_log = []  # train
    EPOCH_log = []   # test

    # client 1 loss
    local_center_loss_log1 = []
    recon_loss_log1 = []
    topo_loss_log1 = []
    total_local_loss_log1 = []

    # client 2 loss
    local_center_loss_log2 = []
    recon_loss_log2 = []
    topo_loss_log2 = []
    total_local_loss_log2 = []

    # client 3 loss
    local_center_loss_log3 = []
    recon_loss_log3 = []
    topo_loss_log3 = []
    total_local_loss_log3 = []

    # global centeredness loss
    global_center_loss_log1 = []
    global_center_loss_log2 = []
    global_center_loss_log3 = []

    # client 1 evaluation
    local_center_eval_log1 = []
    recon_eval_log1 = []

    # client 2 evaluation
    local_center_eval_log2 = []
    recon_eval_log2 = []
    topo_eval_log2 = []

    # client 3 evaluation
    local_center_eval_log3 = []
    recon_eval_log3 = []
    topo_eval_log3 = []

    # global centeredness evaluation
    global_center_eval_log = []

    tick = time.time()
    global_sample1 = None
    global_sample2 = None
    global_sample3 = None
    for epoch in range(total_epochs):
        epoch_log.append(epoch)
        client1.model.train()
        client2.model.train()
        client3.model.train()

        total_losses1 = []
        local_center_losses1 = []
        recon_losses1 = []
        topology_losses1 = []
        global_center_losses1 = []

        total_losses2 = []
        local_center_losses2 = []
        recon_losses2 = []
        topology_losses2 = []
        global_center_losses2 = []

        total_losses3 = []
        local_center_losses3 = []
        recon_losses3 = []
        topology_losses3 = []
        global_center_losses3 = []

        i1_list = []
        i2_list = []
        i3_list = []

        for batch_idx, (data1, data2, data3) in enumerate(zip(X1_train_loader, X2_train_loader, X3_train_loader)):
               
            i1, local_centeredness_loss1, reconstruction_loss1, topology_loss1, global_centeredness_loss1, total_local_loss1 = client1.update_weights(data1, edge_index, global_sample1, sampling_method, freezed1, ablated)
            torch.cuda.empty_cache()
            i2, local_centeredness_loss2, reconstruction_loss2, topology_loss2, global_centeredness_loss2, total_local_loss2 = client2.update_weights(data2, edge_index, global_sample2, sampling_method, freezed2, ablated)
            torch.cuda.empty_cache()
            i3, local_centeredness_loss3, reconstruction_loss3, topology_loss3, global_centeredness_loss3, total_local_loss3 = client3.update_weights(data3, edge_index, global_sample3, sampling_method, freezed3, ablated)
            torch.cuda.empty_cache()
            
            # track normalizer layer1 output
            i1_list.append(i1)
            i2_list.append(i2)
            i3_list.append(i3)

            local_center_losses1.extend(local_centeredness_loss1)
            local_center_losses2.extend(local_centeredness_loss2)
            local_center_losses3.extend(local_centeredness_loss3)

            recon_losses1.extend(reconstruction_loss1)
            recon_losses2.extend(reconstruction_loss2)
            recon_losses3.extend(reconstruction_loss3)

            topology_losses1.extend(topology_loss1)
            topology_losses2.extend(topology_loss2)
            topology_losses3.extend(topology_loss3)

            global_center_losses1.append(global_centeredness_loss1)
            global_center_losses2.append(global_centeredness_loss2)
            global_center_losses3.append(global_centeredness_loss3)

            total_losses1.extend(total_local_loss1)
            total_losses2.extend(total_local_loss2)
            total_losses3.extend(total_local_loss3)
        
        # track global loss
        global_center_loss_log1.append(sum(global_center_losses1)/len(global_center_losses1))
        global_center_loss_log2.append(sum(global_center_losses2)/len(global_center_losses2))
        global_center_loss_log3.append(sum(global_center_losses3)/len(global_center_losses3))

        # track client loss
        local_center_loss_log1.append(sum(local_center_losses1)/len(local_center_losses1))
        local_center_loss_log2.append(sum(local_center_losses2)/len(local_center_losses2))
        local_center_loss_log3.append(sum(local_center_losses3)/len(local_center_losses3))

        recon_loss_log1.append(sum(recon_losses1)/len(recon_losses1))
        recon_loss_log2.append(sum(recon_losses2)/len(recon_losses2))
        recon_loss_log3.append(sum(recon_losses3)/len(recon_losses3))

        topo_loss_log1.append(sum(topology_losses1)/len(topology_losses1))
        topo_loss_log2.append(sum(topology_losses2)/len(topology_losses2))
        topo_loss_log3.append(sum(topology_losses3)/len(topology_losses3))

        total_local_loss_log1.append(sum(total_losses1)/len(total_losses1))
        total_local_loss_log2.append(sum(total_losses2)/len(total_losses2))
        total_local_loss_log3.append(sum(total_losses3)/len(total_losses3))


        if is_federated:
            if epoch % epochs_per_round == 0:
                server.broadcast_weights()
                global_sample1 = server.distribute_representation(1, i1_list, i2_list, i3_list, global_sample_size)
                global_sample2 = server.distribute_representation(2, i1_list, i2_list, i3_list, global_sample_size)
                global_sample3 = server.distribute_representation(3, i1_list, i2_list, i3_list, global_sample_size)

        #Evaluate
        if epoch % 10 == 0: 
            EPOCH_log.append(epoch)
            uni_cbt1, _ = generate_cbt_median(client1.model, X1_train_loader, edge_index)     
            uni_cbt2, _ = generate_cbt_median(client2.model, X2_train_loader, edge_index)
            uni_cbt3, _ = generate_cbt_median(client3.model, X3_train_loader, edge_index)

            # helper.show_image(uni_cbt1.detach().cpu().numpy())
            # helper.show_image(uni_cbt2.detach().cpu().numpy())
            # helper.show_image(uni_cbt3.detach().cpu().numpy())

            # eval local centeredness
            local_center_eval1 = evaluate_local_centeredness(client1.model, X1_test_loader, edge_index, uni_cbt1)
            local_center_eval2 = evaluate_local_centeredness(client2.model, X2_test_loader, edge_index,  uni_cbt2)
            local_center_eval3 = evaluate_local_centeredness(client3.model, X3_test_loader, edge_index,  uni_cbt3)

            local_center_eval_log1.append(local_center_eval1)
            local_center_eval_log2.append(local_center_eval2)
            local_center_eval_log3.append(local_center_eval3)

            # eval reconstruction
            recon_eval1 = evaluate_reconstruction(client1.model, X1_test_loader, edge_index)
            recon_eval2 = evaluate_reconstruction(client2.model, X2_test_loader, edge_index)
            recon_eval3 = evaluate_reconstruction(client3.model, X3_test_loader, edge_index)

            recon_eval_log1.append(recon_eval1)
            recon_eval_log2.append(recon_eval2)
            recon_eval_log3.append(recon_eval3)

            # eval topology

            topo_eval2 = evaluate_topology(client2.model, X2_test, client2.m, edge_index, uni_cbt2)
            topo_eval3 = evaluate_topology(client3.model, X3_test, client3.m, edge_index, uni_cbt3 )

            topo_eval_log2.append(topo_eval2)
            topo_eval_log3.append(topo_eval3)

            # eval global centeredness
            global_center_eval = evaluate_global_centeredness(uni_cbt1, uni_cbt2, uni_cbt3)
            global_center_eval_log.append(global_center_eval)

            tock = time.time()
            time_elapsed = tock - tick
            tick = tock
            local_center_eval1 = float(local_center_eval1)
            test_errors1.append(local_center_eval1)
            local_center_eval2 = float(local_center_eval2)
            test_errors2.append(local_center_eval2)
            local_center_eval3 = float(local_center_eval3)
            test_errors3.append(local_center_eval3)

            print("Epoch: {}  |  Local Centeredness: {:.3f} ; {:.3f} ; {:.3f}  | Reconstruction: {:.3f} ; {:.3f} ; {:.3f} | Topology: Non; {:.3f} ; {:.3f}  |  Global Centeredness: {:.3f} |  Time Elapsed: {:.2f}  |".format(
                epoch, local_center_eval1, local_center_eval2, local_center_eval3, recon_eval1, recon_eval2, recon_eval3, topo_eval2, topo_eval3, global_center_eval, time_elapsed))
            
            #Freeze client control
            if len(test_errors1) > 6:
                torch.save(client1.model.state_dict(), TEMP_FOLDER + "/weight_"+ model_id + "_" + "client1" + "_" + str(local_center_eval1)[:5]  + ".model")
                last_6 = test_errors1[-6:]
                if freeze and not freezed1:
                    if(all(last_6[i] < last_6[i + 1] for i in range(5))):
                        print(f"Freeze model1")
                        freezed1 = True
                        
            
            if len(test_errors2) > 6:
                torch.save(client2.model.state_dict(), TEMP_FOLDER + "/weight_"+ model_id + "_" + "client2" + "_" + str(local_center_eval2)[:5]  + ".model")
                last_6 = test_errors2[-6:]
                if freeze and not freezed2:
                    if(all(last_6[i] < last_6[i + 1] for i in range(5))):
                        print(f"Freeze model2")
                        freezed2 = True

            if len(test_errors3) > 6:
                torch.save(client3.model.state_dict(), TEMP_FOLDER + "/weight_"+ model_id + "_" + "client3" + "_" + str(local_center_eval3)[:5]  + ".model")
                last_6 = test_errors3[-6:]
                if freeze and not freezed3:
                    if(all(last_6[i] < last_6[i + 1] for i in range(5))):
                        print(f"Freeze model3")
                        freezed3 = True
     
    
    # save client training loss
    client1_loss = {'epoch': epoch_log,
                    'local centeredness': local_center_loss_log1,
                    'reconstruction': recon_loss_log1,
                    'topology': topo_loss_log1,
                    'total local loss': total_local_loss_log1,
                    'global centeredness loss': global_center_loss_log1}
    
    client2_loss = {'epoch': epoch_log,
                    'local centeredness': local_center_loss_log2,
                    'reconstruction': recon_loss_log2,
                    'topology': topo_loss_log2,
                    'total local loss': total_local_loss_log2,
                    'global centeredness loss': global_center_loss_log2}
    
    client3_loss = {'epoch': epoch_log,
                    'local centeredness': local_center_loss_log3,
                    'reconstruction': recon_loss_log3,
                    'topology': topo_loss_log3,
                    'total local loss': total_local_loss_log3,
                    'global centeredness loss': global_center_loss_log3}
    
    client_loss = {'client1': client1_loss,
                   'client2': client2_loss,
                   'client3': client3_loss}

    with open(f'{loss_path}/fold{i}_client_loss_log.pkl', 'wb') as file:
                pickle.dump(client_loss, file)
    
    
    # save client test eval
    client1_eval = {'epoch': EPOCH_log,
                    'local_centeredness': local_center_eval_log1,
                    'reconstruction': recon_eval_log1}
    
    client2_eval = {'epoch': EPOCH_log,
                    'local_centeredness': local_center_eval_log2,
                    'reconstruction': recon_eval_log2,
                    'topology': topo_eval2}
    
    client3_eval = {'epoch': EPOCH_log,
                    'local_centeredness': local_center_eval_log3,
                    'reconstruction': recon_eval_log3,
                    'topology': topo_eval_log3}
    
    client_eval = {'client1': client1_eval,
                   'client2': client2_eval,
                   'client3': client3_eval}
    
    with open(f'{eval_path}/fold{i}_client_eval_log.pkl', 'wb') as file:
                pickle.dump(client_eval, file)
    
    # save server test eval
    server_eval = {'epoch': EPOCH_log,
                   'global_centeredness': global_center_eval_log}
    with open(f'{eval_path}/fold{i}_server_eval_log.pkl', 'wb') as file:
                pickle.dump(server_eval, file)

    #Restore best model so far
    try:
        restore1 = "./temp/weight_" + model_id + "_" +  "client1" + "_" + str(min(test_errors1))[:5] + ".model"
        client1.model.load_state_dict(torch.load(restore1))
    except:
        pass

    try:
        restore2 = "./temp/weight_" + model_id + "_" + "client2" + "_" + str(min(test_errors2))[:5] + ".model"
        client2.model.load_state_dict(torch.load(restore2))
    except:
        pass
    
    try:
        restore3 = "./temp/weight_" + model_id + "_" + "client3" + "_" + str(min(test_errors3))[:5] + ".model"
        client3.model.load_state_dict(torch.load(restore3))
    except:
        pass
    torch.save(client1.model.state_dict(), model_path + "client1" + "fold" + str(i) + ".model")
    torch.save(client2.model.state_dict(), model_path + "client2" + "fold" + str(i) + ".model")
    torch.save(client3.model.state_dict(), model_path + "client3" + "fold" + str(i) + ".model")

    # Generate and save refined CBT
    uni_cbt1, embeddings1 = generate_cbt_median(client1.model, X1_train_loader, edge_index) 
    uni_cbt2, embeddings2 = generate_cbt_median(client2.model, X2_train_loader, edge_index)
    uni_cbt3, embeddings3 = generate_cbt_median(client3.model, X3_train_loader, edge_index)

    local_center_eval1 = evaluate_local_centeredness(client1.model, X1_test_loader, edge_index, uni_cbt1)
    local_center_eval2 = evaluate_local_centeredness(client2.model, X2_test_loader, edge_index,  uni_cbt2)
    local_center_eval3 = evaluate_local_centeredness(client3.model, X3_test_loader, edge_index,  uni_cbt3)

    recon_eval1 = evaluate_reconstruction(client1.model, X1_test_loader, edge_index)
    recon_eval2 = evaluate_reconstruction(client2.model, X2_test_loader, edge_index)
    recon_eval3 = evaluate_reconstruction(client3.model, X3_test_loader, edge_index)

    topo_eval2 = evaluate_topology(client2.model, X2_test, client2.m, edge_index, uni_cbt2)
    topo_eval3 = evaluate_topology(client3.model, X3_test, client3.m, edge_index, uni_cbt3 )

    global_center_eval = evaluate_global_centeredness(uni_cbt1, uni_cbt2, uni_cbt3)

    local_center_eval = [local_center_eval1, local_center_eval2, local_center_eval3 ]
    recon_eval = [recon_eval1, recon_eval2, recon_eval3]
    topo_eval = [0, topo_eval2, topo_eval3]
    uni_cbts = [uni_cbt1, uni_cbt2, uni_cbt3]

    uni_cbt1 = uni_cbt1.detach().cpu().numpy()
    np.save(cbt_path1 +  "fold" + str(i) + "_cbt", uni_cbt1)
    uni_cbt2 = uni_cbt2.detach().cpu().numpy()
    np.save(cbt_path2 +  "fold" + str(i) + "_cbt", uni_cbt2)
    uni_cbt3 = uni_cbt3.detach().cpu().numpy()
    np.save(cbt_path3 +  "fold" + str(i) + "_cbt", uni_cbt3)

    embeddings = [embeddings1, embeddings2, embeddings3]
    
    
    print("FINAL RESULTS|  Local Centeredness: {:.5f} ; {:.5f} ; {:.5f}  | Reconstruction: {:.5f} ; {:.5f} ; {:.5f} | Topology: Non ; {:.5f} ; {:.5f} |   Global Centeredness: {:.5f}".format(
                local_center_eval1, local_center_eval2, local_center_eval3, recon_eval1, recon_eval2, recon_eval3, topo_eval2, topo_eval3, global_center_eval))
            
    #Clean interim model weights
    helper.clear_dir(TEMP_FOLDER)
    return uni_cbts, embeddings, local_center_eval, global_center_eval, recon_eval, topo_eval



def k_fold_train(X1, X2, X3):
    model_id = str(uuid.uuid4())

    num_of_subjects = X1.shape[0]

    local_center_folds_nofed1 = []
    local_center_folds_nofed2 = []
    local_center_folds_nofed3 = []
    recon_folds_nofed1 = []
    recon_folds_nofed2 = []
    recon_folds_nofed3 = []
    topo_folds_nofed1 = []
    topo_folds_nofed2 = []
    topo_folds_nofed3 = []
    global_center_folds_nofed = []

    local_center_folds_ablatedfed1 = []
    local_center_folds_ablatedfed2 = []
    local_center_folds_ablatedfed3 = []
    recon_folds_ablatedfed1 = []
    recon_folds_ablatedfed2 = []
    recon_folds_ablatedfed3 = []
    topo_folds_ablatedfed1 = []
    topo_folds_ablatedfed2 = []
    topo_folds_ablatedfed3 = []
    global_center_folds_ablatedfed = []

    local_center_folds_globalfed1 = []
    local_center_folds_globalfed2 = []
    local_center_folds_globalfed3 = []
    recon_folds_globalfed1 = []
    recon_folds_globalfed2 = []
    recon_folds_globalfed3 = []
    topo_folds_globalfed1 = []
    topo_folds_globalfed2 = []
    topo_folds_globalfed3 = []
    global_center_folds_globalfed = []

    for n in range(n_folds):
        torch.cuda.empty_cache()
        print("********* FOLD {} *********".format(n))

        train_indices, test_indices = helper.k_fold_split_indices(num_of_subjects, number_of_folds=n_folds, current_fold_id=n)

        # split trainset and testset
        X1_train = X1[train_indices]
        X1_test = X1[test_indices]

        X2_train = X2[train_indices]
        X2_test = X2[test_indices]

        X3_train = X3[train_indices]
        X3_test = X3[test_indices]

        
        cbts_nofed, embeddings_nofed, local_center_eval_nofed, global_center_eval_nofed, recon_eval_nofed, topo_eval_nofed = train(model_id, n, X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, False, False)
        helper.plot_TSNE(embeddings_nofed, cbts_nofed, 20, n, 'nofed', work_path )
        # helper.plot_PCA(embeddings_nofed, cbts_nofed)
        torch.cuda.empty_cache()
        
        cbts_ablatedfed, embeddings_ablatedfed, local_center_eval_ablatedfed, global_center_eval_ablatedfed, recon_eval_ablatedfed, topo_eval_ablatedfed = train(model_id, n, X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, True, True)
        helper.plot_TSNE(embeddings_ablatedfed, cbts_ablatedfed, 20, n, 'ablatedfed', work_path)
        # helper.plot_PCA(embeddings_fed, cbts_fed)
        torch.cuda.empty_cache() 

        cbts_globalfed, embeddings_globalfed, local_center_eval_globalfed, global_center_eval_globalfed, recon_eval_globalfed, topo_eval_globalfed = train(model_id, n, X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, True, False)
        helper.plot_TSNE(embeddings_globalfed, cbts_globalfed, 20, n, 'globalfed', work_path)
        # helper.plot_PCA(embeddings_fed, cbts_fed)
        torch.cuda.empty_cache() 
        

        local_center_folds_nofed1.append(local_center_eval_nofed[0])
        local_center_folds_nofed2.append(local_center_eval_nofed[1])
        local_center_folds_nofed3.append(local_center_eval_nofed[2])
        recon_folds_nofed1.append(recon_eval_nofed[0])
        recon_folds_nofed2.append(recon_eval_nofed[1])
        recon_folds_nofed3.append(recon_eval_nofed[2])
        topo_folds_nofed1.append(topo_eval_nofed[0])
        topo_folds_nofed2.append(topo_eval_nofed[1])
        topo_folds_nofed3.append(topo_eval_nofed[2])
        global_center_folds_nofed.append(global_center_eval_nofed)

        local_center_folds_globalfed1.append(local_center_eval_globalfed[0])
        local_center_folds_globalfed2.append(local_center_eval_globalfed[1])
        local_center_folds_globalfed3.append(local_center_eval_globalfed[2])
        recon_folds_globalfed1.append(recon_eval_globalfed[0])
        recon_folds_globalfed2.append(recon_eval_globalfed[1])
        recon_folds_globalfed3.append(recon_eval_globalfed[2])
        topo_folds_globalfed1.append(topo_eval_globalfed[0])
        topo_folds_globalfed2.append(topo_eval_globalfed[1])
        topo_folds_globalfed3.append(topo_eval_globalfed[2])
        global_center_folds_globalfed.append(global_center_eval_globalfed)

        local_center_folds_ablatedfed1.append(local_center_eval_ablatedfed[0])
        local_center_folds_ablatedfed2.append(local_center_eval_ablatedfed[1])
        local_center_folds_ablatedfed3.append(local_center_eval_ablatedfed[2])
        recon_folds_ablatedfed1.append(recon_eval_ablatedfed[0])
        recon_folds_ablatedfed2.append(recon_eval_ablatedfed[1])
        recon_folds_ablatedfed3.append(recon_eval_ablatedfed[2])
        topo_folds_ablatedfed1.append(topo_eval_ablatedfed[0])
        topo_folds_ablatedfed2.append(topo_eval_ablatedfed[1])
        topo_folds_ablatedfed3.append(topo_eval_ablatedfed[2])
        global_center_folds_ablatedfed.append(global_center_eval_ablatedfed)


    local_center_nofed1 = sum(local_center_folds_nofed1)/len(local_center_folds_nofed1)
    local_center_nofed2 = sum(local_center_folds_nofed2)/len(local_center_folds_nofed2)
    local_center_nofed3 = sum(local_center_folds_nofed3)/len(local_center_folds_nofed3)
    recon_nofed1 = sum(recon_folds_nofed1)/len(recon_folds_nofed1)
    recon_nofed2 = sum(recon_folds_nofed2)/len(recon_folds_nofed2)
    recon_nofed3 = sum(recon_folds_nofed3)/len(recon_folds_nofed3)
    topo_nofed1 = sum(topo_folds_nofed1)/len(topo_folds_nofed1)
    topo_nofed2 = sum(topo_folds_nofed2)/len(topo_folds_nofed2)
    topo_nofed3 = sum(topo_folds_nofed3)/len(topo_folds_nofed3)
    global_center_nofed = sum(global_center_folds_nofed)/len(global_center_folds_nofed)

    local_center_globalfed1 = sum(local_center_folds_globalfed1)/len(local_center_folds_globalfed1)
    local_center_globalfed2 = sum(local_center_folds_globalfed2)/len(local_center_folds_globalfed2)
    local_center_globalfed3 = sum(local_center_folds_globalfed3)/len(local_center_folds_globalfed3)
    recon_globalfed1 = sum(recon_folds_globalfed1)/len(recon_folds_globalfed1)
    recon_globalfed2 = sum(recon_folds_globalfed2)/len(recon_folds_globalfed2)
    recon_globalfed3 = sum(recon_folds_globalfed3)/len(recon_folds_globalfed3)
    topo_globalfed1 = sum(topo_folds_globalfed1)/len(topo_folds_globalfed1)
    topo_globalfed2 = sum(topo_folds_globalfed2)/len(topo_folds_globalfed2)
    topo_globalfed3 = sum(topo_folds_globalfed3)/len(topo_folds_globalfed3)
    global_center_globalfed = sum(global_center_folds_globalfed)/len(global_center_folds_globalfed)

    local_center_ablatedfed1 = sum(local_center_folds_ablatedfed1)/len(local_center_folds_ablatedfed1)
    local_center_ablatedfed2 = sum(local_center_folds_ablatedfed2)/len(local_center_folds_ablatedfed2)
    local_center_ablatedfed3 = sum(local_center_folds_ablatedfed3)/len(local_center_folds_ablatedfed3)
    recon_ablatedfed1 = sum(recon_folds_ablatedfed1)/len(recon_folds_ablatedfed1)
    recon_ablatedfed2 = sum(recon_folds_ablatedfed2)/len(recon_folds_ablatedfed2)
    recon_ablatedfed3 = sum(recon_folds_ablatedfed3)/len(recon_folds_ablatedfed3)
    topo_ablatedfed1 = sum(topo_folds_ablatedfed1)/len(topo_folds_ablatedfed1)
    topo_ablatedfed2 = sum(topo_folds_ablatedfed2)/len(topo_folds_ablatedfed2)
    topo_ablatedfed3 = sum(topo_folds_ablatedfed3)/len(topo_folds_ablatedfed3)
    global_center_ablatedfed = sum(global_center_folds_ablatedfed)/len(global_center_folds_ablatedfed)
    
    plot_results = pd.DataFrame({"not federated":[local_center_nofed1, local_center_nofed2, local_center_nofed3, global_center_nofed], 
                                  "global federated": [local_center_globalfed1, local_center_globalfed2, local_center_globalfed3, global_center_globalfed],
                                  "ablated federated": [local_center_ablatedfed1, local_center_ablatedfed2, local_center_ablatedfed3, global_center_ablatedfed]},
                                  index = ["client1", "client2", "client3", "global centeredness"])
    # final_results = pd.DataFrame({"federated":[local_center_fed1, local_center_fed2, local_center_fed3]},
    #                               index = ["client 1", "client2", "client3"])
    recon_results = pd.DataFrame({"not federated":[recon_nofed1, recon_nofed2, recon_nofed3],
                                  "global federated": [ recon_globalfed1, recon_globalfed2, recon_globalfed3],
                                  "ablated federated": [recon_ablatedfed1, recon_ablatedfed2, recon_ablatedfed3]},
                                  index = ["client1", "client2", "client3"])
    
    topo_results = pd.DataFrame({"not federated":[topo_nofed1, topo_nofed2, topo_nofed3],
                                  "global federated": [ topo_globalfed1, topo_globalfed2, topo_globalfed3],
                                  "ablated federated": [topo_ablatedfed1, topo_ablatedfed2, topo_ablatedfed3]},
                                  index = ["client1", "client2", "client3"])
    return plot_results, recon_results, topo_results




        
    



