from helper import show_image
import numpy as np
from config import CONFIG
import os
import matplotlib.pyplot as plt
import pickle

def plot_CBT(model_name, sampling_method):
    save_path = f'./output/{sampling_method}/{model_name}/cbt_globalfed'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cbt_path = f'./output/{sampling_method}/{model_name}/global_fed'
    for n in range(3):
        for i in range(4):
            print("*********client {}   fold {} *********".format(n+1, i))
            cbt = np.load(f'{cbt_path}/client{n+1}_cbts/fold{i}_cbt.npy')
            show_image(cbt, n+1, i, save_path)



def plot_client_training_log(client_name, loss_data_nofed, loss_data_fed, fold_num, sampling_method, save_path):
    print("********* {}   fold {} *********".format(client_name, fold_num))

    fig, axs = plt.subplots(1, 5, figsize = (20,4))
    for i, key in enumerate(["local centeredness", "reconstruction", "topology", "total local loss", "global centeredness loss"]):
        loss_fed = loss_data_fed[key]
        loss_nofed = loss_data_nofed[key]
        epoch = loss_data_fed["epoch"]
        axs[i].plot(epoch, loss_fed, 'tab:orange', label='Federated')  # Add label for federated loss
        axs[i].plot(epoch, loss_nofed, 'tab:green', label='Non-federated')  # Add label for non-federated loss
        axs[i].set(xlabel= 'epoch', ylabel= f'{key} loss')
        axs[i].set_title(f'{key} loss')
        axs[i].legend()  # Show the legend
    plt.suptitle(f'fold {fold_num} of {client_name} with {sampling_method} sampling on train set')
    plt.savefig(f'{save_path}/{client_name}_trainloss_fold{fold_num}')
    plt.show()



def plot_training_log(model_name, sampling_method, n_folds):

    save_path = f'./output/{sampling_method}/{model_name}/client_loss_plot'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for client in ['client1', 'client2', 'client3']:
        for n in range(n_folds):
            with open(f'./output/{sampling_method}/{model_name}/nofed/loss/fold{n}_client_loss_log.pkl', 'rb') as file:
                log_nofed = pickle.load(file)
            with open(f'./output/{sampling_method}/{model_name}/global_fed/loss/fold{n}_client_loss_log.pkl', 'rb') as file:
                log_fed = pickle.load(file)
            plot_client_training_log(client, log_nofed[client], log_fed[client], n, sampling_method, save_path)


def plot_client_eval_log(client_name, eval_data_nofed, eval_data_fed, fold_num, sampling_method, save_path):
    print("********* {}   fold {} *********".format(client_name, fold_num))

    fig, axs = plt.subplots(1, 1, figsize = (6,4))
    for i, key in enumerate(["local_centeredness"]):
        loss_fed = eval_data_fed[key]
        loss_nofed = eval_data_nofed[key]
        epoch = eval_data_fed["epoch"]
        print(len(epoch))
        print(len(loss_fed))
        axs.plot(epoch, loss_fed, 'tab:orange', label='Federated')  # Add label for federated loss
        axs.plot(epoch, loss_nofed, 'tab:green', label='Non-federated')  # Add label for non-federated loss
        axs.set(xlabel= 'epoch', ylabel= f'{key} loss')
        axs.set_title(f'{key} loss')
        axs.legend()  # Show the legend
    plt.suptitle(f'fold {fold_num} of {client_name} with {sampling_method} sampling on train set')
    plt.savefig(f'{save_path}/{client_name}_trainloss_fold{fold_num}')
    plt.show()



def plot_eval_log(model_name, sampling_method, n_folds):

    save_path = f'./output/{sampling_method}/{model_name}/client_eval_plot'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for client in ['client1', 'client2', 'client3']:
        for n in range(n_folds):
            with open(f'./output/{sampling_method}/{model_name}/nofed/eval/fold{n}_client_eval_log.pkl', 'rb') as file:
                log_nofed = pickle.load(file)
            with open(f'./output/{sampling_method}/{model_name}/global_fed/eval/fold{n}_client_eval_log.pkl', 'rb') as file:
                log_fed = pickle.load(file)
            plot_client_eval_log(client, log_nofed[client], log_fed[client], n, sampling_method, save_path)


if __name__ ==  "__main__":
    sampling_method = CONFIG['sampling_method']
    model_name = CONFIG['model_name']
    n_folds = CONFIG['n_folds']


    # plot_CBT(model_name, sampling_method)
    # plot_training_log(model_name, sampling_method, n_folds)
    plot_eval_log(model_name, sampling_method, n_folds)