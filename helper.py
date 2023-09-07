import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import KFold
import torch
import random
import os
import torch.utils.data as data_utils
import torch
from sklearn.decomposition import PCA

np.random.seed(35813)
torch.manual_seed(35813)
# def free_gpu_cache():
#     # print("Initial GPU Usage")
#     # gpu_usage()                             

#     torch.cuda.empty_cache()

#     cuda.select_device(0)
#     cuda.close()
#     cuda.select_device(0)

#     # print("GPU Usage after emptying the cache")
#     # gpu_usage()

#Clears the given directory
def clear_dir(dir_name):
    for file in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file))

def k_fold_split_indices(length_of_data, number_of_folds, current_fold_id):
    """split indice of fold n in k-fold

    Args:
        len_of_data (int)
        num_of_folds (_type_)
        current_fold_id (_type_)
    
    Returns:
        list: indices of trainset
        list: indices pf testset
        
    """
    kf = KFold(n_splits=number_of_folds,shuffle=True, random_state= 42)
    split_indices = kf.split(range(length_of_data))
    train_indices, test_indices = [(list(train), list(test)) for train, test in split_indices][current_fold_id]
    return train_indices, test_indices
 
def get_loader(feature_vectors, batch_size, num_workers=1):
    """
    Build and return a data loader.
    """
    loader = data_utils.DataLoader(feature_vectors.astype(np.float32),
                        batch_size=batch_size,
                        shuffle = True, #set to True in case of training and False when testing the model
                        num_workers=num_workers
                        )
    
    return loader

def get_edge_index(batch_size, device):
    edge_index = torch.zeros(2, batch_size)
    for i in range(batch_size):
        edge_index[:, i] = torch.tensor([i, i])
    return edge_index.clone().long().to(device)

def pre_horizontal_antiVectorize(vec, m):
    """
    Anti-vectorize Feature vector [1, m*(m-1)/2] -> [m, m]
    The input and output are both numpy.array
    """
    assert vec.shape[0] == m * (m - 1) / 2, "vec must be of length m*(m-1)/2 i.e. it does not contain the diagonal entries"
    
    M = np.zeros((m, m))
    M[np.triu_indices(m, k= 1)] = vec
    M = M + M.T
    return M

def pre_vertical_vectorize(M):
    """
    The input and output are both numpy.array
    Vectorize a square matrix M into a 1D numpy array
    without diagonal entries.
    """
    m = M.shape[0]
    assert M.shape == (m, m), "M must be a square matrix"
    vec = M[np.tril_indices(m, k= -1)]
    return vec

def antiVectorize(vec, m):  # vertical
    """Anti-vectorize Feature vector [1, m*(m-1)/2] -> [m, m]
       The input and output are both torch.tensor
    """
    assert vec.shape[1] == m * (m - 1) // 2, "vec must be of length m*(m-1)/2 i.e. it does not contain the diagonal entries"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    M = torch.zeros((m, m), dtype=torch.float32, device=device)
    tril_indices = torch.tril_indices(m, m, -1, device=device)
    M[tril_indices[0], tril_indices[1]] = vec.to(device=device, dtype=torch.float32).squeeze()
    M = M + M.t()
    return M

def batch_antiVectorize(vec, m):  # vertical
    """
    The input and output are both torch.tensor. can process more than 1
    Convert 2D feature vector [batch_size, m*(m-1)/2] to 3D matrix [batch_size, m, m]
    """
    assert vec.shape[1] == m * (m - 1) // 2, "vec must be of length m*(m-1)/2 i.e. it does not contain the diagonal entries"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = vec.shape[0]
    M = torch.zeros((batch_size, m, m), dtype=torch.float32, device=device)
    tril_indices = torch.tril_indices(m, m, -1)
    M[:, tril_indices[0], tril_indices[1]] = vec.to(device=device, dtype=torch.float32).squeeze()
    M = M + M.transpose(1, 2)
    return M

def k_means_clustering_sampling(feature_matrix, n_clusters, sample_size):
    """The feature matrix should be a numpy array
    """
    
    # number of subjects
    n_subjects = np.shape(feature_matrix)[0]

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=2)
    kmeans.fit(feature_matrix)

    # Get the cluster labels for each subject
    cluster_labels = kmeans.labels_

    # Create a DataFrame to hold the data and cluster labels
    data = pd.DataFrame({'subject_id': np.arange(0, n_subjects),  
                        'cluster_label': cluster_labels})

    # Determine sample size per cluster
    total_sample_size = sample_size  # Total number of subjects to sample
    sample_size_per_cluster = total_sample_size // n_clusters  # Equal sample size per cluster

    # Perform sampling
    # Group the data by cluster label and sample subjects from each cluster
    cluster_samples = data.groupby('cluster_label').apply(lambda x: x.sample(n=sample_size_per_cluster, replace=True))

    # Use the sampled subjects for further analysis or model training
    sampled_subjects_id = cluster_samples['subject_id'].values

    return sampled_subjects_id.tolist()


def hierarchical_clustering_sampling(feature_matrix, n_clusters, sample_size):
    
    # Cluster the data
    hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = hierarchical_clustering.fit_predict(feature_matrix)  # Get the cluster labels for each subject

    # Sample subjects from clusters
    sampled_subjects_id = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]  # Get indices of subjects in the current cluster
        if len(cluster_indices) > 0:
            # Randomly sample one subject from the current cluster
            sampled_subject = random.choice(cluster_indices)
            sampled_subjects_id.append(sampled_subject)

    # Sample additional subjects if needed to reach a total of 10 sampled subjects
    while len(sampled_subjects_id) < sample_size:
        # Randomly sample one subject from any cluster
        cluster = random.randint(0, n_clusters - 1)
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) > 0:
            sampled_subject = random.choice(cluster_indices)
        sampled_subjects_id.append(sampled_subject)
    
    return sampled_subjects_id


def random_sampling(feature_matrix, sample_size):

    n_subjects =  np.shape(feature_matrix)[0]
    subject_ids = np.arange(1, n_subjects)
    sampled_subjects_id = np.random.choice(subject_ids, size=sample_size, replace=False)

    return sampled_subjects_id.tolist()


    
# def show_image(img):
#     """The input should be numpy.array
#         i is fold number
#     """
#     # img = np.repeat(np.repeat(img, 10, axis=1), 10, axis=0)
#     plt.imshow(img)
#     plt.colorbar()
#     plt.axis('on')
#     plt.show()

def show_image(img, n, i, save_path):
    """The input should be numpy.array
        i is fold number
    """
    # img = np.repeat(np.repeat(img, 10, axis=1), 10, axis=0)
    plt.imshow(img)
    plt.colorbar()
    plt.title("fold " + str(i))
    plt.axis('on')
    plt.savefig(f'{save_path}/client{n}_fold{i}_cbt', dpi=300)
    plt.show()


def batch_vectorize(M):
    """
    Convert a batch of square matrices M of shape [batch_size, m, m]
    into a batch of vectorized feature tensors of shape [batch_size, m*(m-1)/2].
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, m, _ = M.shape
    assert M.shape[1] == M.shape[2], "M must be a square matrix"

    vec = M[:, np.tril_indices(m, k=-1)[0], np.tril_indices(m, k=-1)[1]]
    vec = vec.view(batch_size, -1)

    return vec.detach().cpu().numpy()

def vectorize(M):   # vertical
    """
    Vectorize a square matrix M into a 1D numpy array
    without diagonal entries.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = M.shape[0]
    assert M.shape == (m, m), "M must be a square matrix"
    vec = M[np.tril_indices(m, k=-1)]
    return vec.clone().detach().requires_grad_(False).reshape(1, -1).to(device=device)


def plot_TSNE(embeddings_list, cbt_list, perplexity, i, type, save_path):
    vectorized_embeddings = [batch_vectorize(embedding) for embedding in embeddings_list]
    all_embeddings = np.concatenate(vectorized_embeddings , axis=0)
    vectorized_cbts = [vectorize(cbt).cpu().numpy() for cbt in cbt_list]
    all_cbts = np.concatenate(vectorized_cbts, axis=0)
    plot_data = np.concatenate((all_embeddings, all_cbts), axis = 0)
    # Define colors and labels for the embeddings
    colors = ['lightcoral', 'lightsteelblue', 'lime', 'red', 'blue', 'green']
    labels = ['Embeddings of hospital 1', 'Embeddings of hospital 2', 'Embeddings of hospital 3', 'Hospital1-specific CBT', 'Hospital2-specific CBT', 'Hospital3-specific CBT']
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)

    # Transform the embeddings to 2D for visualization
    all_embeddings_2d = tsne.fit_transform(plot_data)

    # Plot the transformed embeddings with colors
    plt.figure(figsize=(8, 8))

    # Iterate over embeddings and plot them with corresponding color and label
    start = 0
    for idx, length in enumerate([len(embeddings_list[0]), len(embeddings_list[1]), len(embeddings_list[2]), 1, 1, 1]):
        end = start + length
        plt.scatter(all_embeddings_2d[start:end, 0], all_embeddings_2d[start:end, 1], c=colors[idx], label=labels[idx])
        start = end

    plt.title('t-SNE visualization of all embeddings and client-based CBT')
    plt.legend()  # Add a legend
    plt.savefig(f'{save_path}/fold{i}_{type}_tsne')
    plt.show()


def plot_PCA(embeddings_list, cbt_list):
    vectorized_embeddings = [batch_vectorize(embedding) for embedding in embeddings_list]
    all_embeddings = np.concatenate(vectorized_embeddings , axis=0)
    vectorized_cbts = [vectorize(cbt).cpu().numpy() for cbt in cbt_list]
    all_cbts = np.concatenate(vectorized_cbts, axis=0)
    plot_data = np.concatenate((all_embeddings, all_cbts), axis = 0)

    # Create color labels for the embeddings
    colors = ['lightcoral'] * len(embeddings_list[0]) + ['lightsteelblue'] * len(embeddings_list[1]) + ['lime'] * len(embeddings_list[2]) + ["red"] + ["blue"] +["green"]
    labels = ['Embeddings of hospital 1', 'Embeddings of hospital 2', 'Embeddings of hospital 3', 'Hospital1-specific CBT', 'Hospital2-specific CBT', 'Hospital3-specific CBT']  # Label each category

    pca = PCA(n_components=2)

    # Transform the embeddings to 2D for visualization
    all_embeddings_2d = pca.fit_transform(plot_data)

    # Plot the transformed embeddings with colors
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(all_embeddings_2d[:, 0], all_embeddings_2d[:, 1], c=colors)
    plt.title('PCA visualization of all embeddings')

    # Create a legend for the scatter plot
    handles, _ = scatter.legend_elements()
    plt.legend(handles, labels, title="Categories")

    # plt.savefig(f'{save_path}/client{n}_fold{i}_cbt')
    plt.show()

def plot_final_eval_metric(plotdata):
    plotdata.plot(kind="bar",figsize=(15, 8))

    plt.title("Centeredness of CBT on test set")

    plt.ylabel("Frob distance between embedded MRIs and their CBT")