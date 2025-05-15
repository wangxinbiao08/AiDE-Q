import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from net.Qtype_net import Qtype
from net.Qtype_datasetloader import Dataset_Qtype, DatasetLoader_Qtype


def Qtype_val(dataset_path_list,
              batch_size,
              pretrain_model_path, device="cuda",
              **param):
    Net = Qtype(**param).to(device)
    Net.load_state_dict(torch.load(pretrain_model_path, map_location=device))
    criterion = nn.MSELoss().to(device)

    differences = []

    for idx, (n_qubits, bits, recipes, renyi_Entropy_3q) in enumerate(DatasetLoader_Qtype(dataset_path_list, batch_size)):
        input_data = (bits + recipes * 2).to(device).float()
        input_data = input_data[:, :, 0:2500]
        input_data = input_data.repeat(1, 1, 2)
        input_data = input_data.swapaxes(1, 2)
        output = Net(input_data)

        target_tensor = renyi_Entropy_3q.clone().detach().to(dtype=output.dtype, device=device)
        loss = criterion(output, target_tensor)
        difference = output.cpu().detach() - target_tensor.cpu().detach()
        differences.append(difference.view(-1).numpy())
    differences = np.concatenate(differences)
    bins = 50
    hist, bin_edges = np.histogram(differences, bins=bins)
    # 计算每个区间的中心点（即每个区间的中值，用来作为横坐标）
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), color='blue', alpha=0.7)

    # 添加图形标签和标题
    plt.xlabel('Difference between output and target', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Differences between Output and Target', fontsize=14)
    plt.grid(True)
    plt.show()







