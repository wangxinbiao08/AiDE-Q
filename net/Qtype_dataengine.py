import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from net.Qtype_net import Qtype
from net.Qtype_datasetloader import Dataset_Qtype, DatasetLoader_Qtype
from utils.utils import expand_data
import os
import shutil
import h5py


def Qtype_dataengine(evaluation_dataset_path, remaining_dataset_path, logs_save_path,
                     batch_size, dataengine_shots, physic_condition,
                     subset_num, subset_precentage, select_percentage,
                     pretrain_model_path,  device="cuda:5",
                     **param):

    logs = f'------------data-engine start optimization----------------\n'
    with open(logs_save_path, 'a') as file:
        file.write(logs)

    shots = param.get('shots', 5000)
    n_qubits = param.get('n_qubits', 50)
    Net = Qtype(**param).to(device)
    Net.eval()
    Net.load_state_dict(torch.load(pretrain_model_path, map_location=device))

    


    collected_var_per_records = []   # collect confidence for the records in each batch
    collected_all_bits = []
    collected_all_recipes = []
    collected_all_renyi_Entropy = []
    collected_all_param_J = []
    collected_all_param_g = []
    
    for idx, (n_qubits_ls, bits, recipes, renyi_Entropy_3q, param_J, param_g) in enumerate(DatasetLoader_Qtype([remaining_dataset_path], batch_size)):
        current_batch_size = bits.size(0)
        
        bits = expand_data(bits[:, :, :dataengine_shots], shots)
        recipes = expand_data(recipes[:, :, :dataengine_shots], shots)

        if physic_condition == True:
            combined_param = torch.stack([param_J.float(), param_g.float()], dim=1).float().to(device)
        else:
            combined_param = torch.stack([0,0], dim=1).float().to(device)
        collected_all_bits.append(bits)
        collected_all_recipes.append(recipes)
        collected_all_renyi_Entropy.append(renyi_Entropy_3q)
        collected_all_param_J.append(param_J)
        collected_all_param_g.append(param_g)
        
        # collect the prediction results of all subsets
        all_output = np.zeros((subset_num, current_batch_size, n_qubits_ls[0]))
        for s in range(subset_num):
            subset_size = int(dataengine_shots * subset_precentage)
            indices = torch.randint(0, dataengine_shots, (subset_size,))
            input_data = (bits + recipes * 2).to(device).float()
            evaluation_data = input_data[:, :, indices]
            result_data = expand_data(evaluation_data, shots)
            # result_data = evaluation_data.repeat(1, 1, int(1 / subset_precentage)).to(device)
            result_data = result_data.swapaxes(1, 2)
            output, _ = Net(result_data, combined_param)
            all_output[s] = output.cpu().detach().numpy()

        var_per_qubits = np.var(all_output, axis=0)
        var_per_records = np.mean(var_per_qubits, axis=1)  # confidence metric
        var_per_records = torch.from_numpy(var_per_records)
        collected_var_per_records.append(var_per_records)
    
    
    # Concatenate the data of all batches
    collected_var_per_records = torch.cat(collected_var_per_records, dim=0)
    collected_all_bits = torch.cat(collected_all_bits, dim=0)
    collected_all_recipes = torch.cat(collected_all_recipes, dim=0)
    collected_all_renyi_Entropy =  torch.cat(collected_all_renyi_Entropy, dim=0)
    collected_all_param_J = torch.cat(collected_all_param_J, dim=0)
    collected_all_param_g = torch.cat(collected_all_param_g, dim=0)


    # Extract the index with small variance 
    sorted_indices = np.argsort(collected_var_per_records.numpy())
    min_variance_indices = sorted_indices[:max(1, int(collected_all_bits.size(0) * select_percentage))]
    
    # Extract the data with small variance 
    collected_evaluation_bits = collected_all_bits[min_variance_indices, :, :]
    collected_evaluation_recipes = collected_all_recipes[min_variance_indices, :, :]
    c_input_data = (collected_evaluation_bits + collected_evaluation_recipes * 2).to(device).float()
    c_input_data = c_input_data[:, :, 0:shots]
    c_input_data = c_input_data.swapaxes(1, 2)

    collected_evaluation_param_J = collected_all_param_J[min_variance_indices]
    collected_evaluation_param_g = collected_all_param_g[min_variance_indices]


    c_combined_param = torch.stack([collected_evaluation_param_J.float(), collected_evaluation_param_g.float()], dim=1).float().to(device)

    c_output, _ = Net(c_input_data, c_combined_param)
    collected_evaluation_renyi_Entropy = c_output.cpu().detach()
        
    # Obtaining remaining indices
    remaining_indices = np.setdiff1d(np.arange(collected_all_bits.size(0)), min_variance_indices)
    # Obtaining the data related to the remaining indices
    collected_remaining_bits = collected_all_bits[remaining_indices, :, :]
    collected_remaining_recipes = collected_all_recipes[remaining_indices, :, :]
    collected_remaining_renyi_Entropy = collected_all_renyi_Entropy[remaining_indices, :]
    collected_remaining_param_J = collected_all_param_J[remaining_indices]
    collected_remaining_param_g = collected_all_param_g[remaining_indices]

    # Save all data
    if os.path.exists(evaluation_dataset_path):
        os.remove(evaluation_dataset_path)
    with h5py.File(evaluation_dataset_path, 'a') as f:
        for record_index, (bits, recipes, renyi_Entropy_3q, param_J, param_g) in enumerate(
                zip(collected_evaluation_bits, collected_evaluation_recipes, collected_evaluation_renyi_Entropy, collected_evaluation_param_J, collected_evaluation_param_g)):
            record_group = f.create_group(f'record_{record_index}')
            record_group.create_dataset('n_qubit', data=n_qubits_ls[0])
            record_group.create_dataset('bits', data=bits)
            record_group.create_dataset('recipes', data=recipes)
            record_group.create_dataset('renyi_Entropy_3q', data=renyi_Entropy_3q)
            record_group.create_dataset('param_J', data=param_J)
            record_group.create_dataset('param_g', data=param_g)
            

    if os.path.exists(remaining_dataset_path):
        os.remove(remaining_dataset_path)
    with h5py.File(remaining_dataset_path, 'w') as f:  
        for record_index, (bits, recipes, renyi_Entropy_3q, param_J, param_g) in enumerate(
                zip(collected_remaining_bits, collected_remaining_recipes, collected_remaining_renyi_Entropy, collected_remaining_param_J, collected_remaining_param_g)):
            record_group = f.create_group(f'record_{record_index}')
            record_group.create_dataset('n_qubit', data=n_qubits_ls[0])
            record_group.create_dataset('bits', data=bits)
            record_group.create_dataset('recipes', data=recipes)
            record_group.create_dataset('renyi_Entropy_3q', data=renyi_Entropy_3q)
            record_group.create_dataset('param_J', data=param_J)
            record_group.create_dataset('param_g', data=param_g)

