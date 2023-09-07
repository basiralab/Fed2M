import numpy as np
import os


def simulate_data(subject_size):
    # Simulate data
    roi35 = np.random.randn(subject_size, 595)
    roi160 = np.random.randn(subject_size, 12721)
    roi1268 = np.random.randn(subject_size, 35778)

    # Save to numpy files
    save_path = f'./simulated_dataset'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save("simulated_dataset/roi35.npy", roi35)
    np.save("simulated_dataset/roi160.npy", roi160)
    np.save("simulated_dataset/roi1268.npy", roi1268)


if __name__ == "__main__":

    subject_size = 200  #to be edit

    simulate_data(subject_size)
