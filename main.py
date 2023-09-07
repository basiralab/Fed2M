import torch
import scipy.io
import numpy as np


'''
Setting up working environment
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device} is ready.')


'''
setting up working directory
'''
content_path = "/home/jia/Desktop/MSc_Project/Fed2M"
data_path = "/home/jia/Desktop/MSc_Project/m2integrator/Fed2M"




# SLIM data
# '''
# Import Data (Load .mat file)
# '''
# # Feature vector 1
# roi_35_o = scipy.io.loadmat(f'{data_path}/morph_thickness_data_35_avg.mat')['morph_thickness_data_35_avg']

# # roi_35 = scipy.io.loadmat(f'{data_path}/morph_thickness_data_35_avg.mat')['morph_thickness_data_35_avg']
# shape1 = roi_35_o.shape
# print(f'shape of dataset1 is {shape1}')
# # Feature vector2
# roi_160 = scipy.io.loadmat(f'{data_path}/func_data_160.mat')['LR']

# # roi_160 = scipy.io.loadmat(f'{data_path}/func_data_160.mat')['LR']
# shape2 = roi_160.shape
# print(f'shape of dataset2 is  {shape2}')
# # Feature vector3
# roi_268 = scipy.io.loadmat(f'{data_path}/func_data_268.mat')['HR']

# # roi_268 = scipy.io.loadmat(f'{data_path}/func_data_268.mat')['HR']
# shape3 = roi_268.shape
# print(f'shape of dataset3 is {shape3}')


# '''
# Load .mat file
# '''
# #By visualising the dataset, we found mophological feature vectors were vectorized  horizontally, but functional feature vectors were 
# #vectorized vertically. Therefore, we firstly formallize all the data in a **vertically-verctorized** form.

# #- Morphological data : roi_35
# #- Functional data: roi_160 and roi_268

# import helper
# roi_35 = []
# for data in roi_35_o:
#     data = helper.pre_horizontal_antiVectorize(data, 35)
#     data = helper.pre_vertical_vectorize(data)
#     roi_35.append(data)

# roi_35 = np.array(roi_35)
    
# #It is worth to mention that functional MRI should not have negative values, so we 'diacard' all negative values in fucntional MRI, which means set all negative values to zero.
# roi_160 = np.clip(roi_160, a_min=0, a_max=None)
# roi_268 = np.clip(roi_268, a_min=0, a_max=None)



# Simulated dataset
roi_35 = np.load(f'./simulated_dataset/roi_35.npy')
roi_160 = np.load(f'./simulated_dataset/roi_160.npy')
roi_268 = np.load(f'./simulated_dataset/roi_268.npy')


''''
Train Fed2M
'''

from demo import k_fold_train
final_reults, recon_results, topo_results = k_fold_train(roi_35, roi_160, roi_268)


'''
Visualise evaluation metric
'''
from helper import plot_final_eval_metric

from config import CONFIG

print("--------Local results---------")
print(final_reults)
print("--------Recon results---------")
print(recon_results)
print("--------Topo results---------")
print(topo_results)


plot_final_eval_metric(final_reults)

plot_final_eval_metric(recon_results)

plot_final_eval_metric(topo_results)
