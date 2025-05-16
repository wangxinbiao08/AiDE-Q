import h5py
import numpy as np
import torch

import os
from utils.prob_cal import cal_entropy



def expand_data_2D(data, shots):
    h = data.shape[-1]

    if h > shots:
        raise ValueError("the dimension of the data should be less than shots")

    expanded_data = np.tile(data, (1, shots // h + 1))

    if expanded_data.shape[-1] > shots:
        expanded_data = expanded_data[:, :shots]

    return expanded_data


def split_datasets(dataset_paths,
                   train_dataset_save_path, val_dataset_save_path, dataengine_dataset_save_path,
                   train_records_num, val_records_num, dataengine_records_num, p_noise, shots, small_shots=None):
    records = []

    # read all records
    entropy = []
    with h5py.File(dataset_paths, 'r') as file:
        for group in file.values():
            record = {}
            for name, data in group.items():
                # check if the data is a scalar 
                if data.shape == ():
                    record[name] = data[()] 
                else:
                    record[name] = data[:]  
                if name == 'renyi_Entropy_3q':
                    entropy.append(data[:])
            records.append(record)
    entropy = np.array(entropy)
    mean_entropy = np.mean(entropy, axis=0)

    # randomly selecting records to construct hybrid datasets
    np.random.shuffle(records)
    total_records_num = train_records_num + val_records_num + dataengine_records_num
    train_records = records[:train_records_num]
    val_records = records[train_records_num:train_records_num+val_records_num]
    dataengine_records = records[train_records_num+val_records_num:total_records_num]
    print(f't_num: {train_records_num}, val_num: {val_records_num}, records_num: {len(records)}')
    print(f"total: {total_records_num}, train: {len(train_records)}, val: {len(val_records)}, de: {len(dataengine_records)}")

    # save training dataset 
    noise_data_idx_ls = np.random.choice(train_records_num, size=int(train_records_num*p_noise), replace=False)  # noise set 0 for DE_engine
    
    with h5py.File(train_dataset_save_path, 'w') as train_file:
        for idx, record in enumerate(train_records):
            group = train_file.create_group(f'record_{idx}')
            if idx in noise_data_idx_ls:  #< p_noise:   # noise should be set as 0 for DE_engine
                input_shots = small_shots
            else:
                input_shots = shots
            for key, data in record.items():
                if key == 'renyi_Entropy_3q':
                    noise = np.random.normal(0, 1/(input_shots ** 0.5), size=data.shape)
                    # noise = np.random.normal(0, 0.2, size=data.shape)
                    exp_factor = [2**(i+1) for i in range(3)]
                    exp_factor.extend([2**3 for i in range(data.shape[0]-3)])
                    data = data + noise * mean_entropy * np.array(exp_factor)
                if key in ['bits', 'recipes']:
                    data = data[:, :input_shots]
                    if input_shots < shots:
                        data = expand_data_2D(data, shots)
                group.create_dataset(key, data=data)

    # Save the validation dataset
    with h5py.File(val_dataset_save_path, 'w') as val_file:
        for idx, record in enumerate(val_records):
            group = val_file.create_group(f'record_{idx}')
            for key, data in record.items():
                if key in ['bits', 'recipes']:
                    data = data[:, :shots]
                group.create_dataset(key, data=data)


    # Save the remaining dataset
    with h5py.File(dataengine_dataset_save_path, 'w') as engine_file:
        for idx, record in enumerate(dataengine_records):
            group = engine_file.create_group(f'record_{idx}')
            for key, data in record.items():
                if key in ['bits', 'recipes']:
                    data = data[:, :shots]
                group.create_dataset(key, data=data)

def mend_datasets(dataset_paths, save_path):
    records = []

    # read all records
    for path in dataset_paths:
        with h5py.File(path, 'r') as file:
            for group in file.values():
                record = {}
                for name, data in group.items():
                    # check if data is scalar
                    if data.shape == ():
                        record[name] = data[()]  
                    else:
                        record[name] = data[:]  
                records.append(record)


    np.random.shuffle(records)
    
    if os.path.exists(save_path):
        os.remove(save_path)

    # Save the training dataset
    with h5py.File(save_path, 'w') as train_file:
        for idx, record in enumerate(records):
            group = train_file.create_group(f'record_{idx}')
            for key, data in record.items():
                group.create_dataset(key, data=data)


def reallocate_datasets(evaluation_dataset_path, remaining_dataset_path=None):
    evaluation_records = []
    remaining_records = []
    # read all records
    with h5py.File(evaluation_dataset_path, 'r') as file:
        evaluation_dataset_num = len(file.keys())
        for j, group in enumerate(file.values()):
            record = {}
            for name, data in group.items():
                # check if data is scalar
                if data.shape == ():
                    record[name] = data[()]  
                else:
                    record[name] = data[:]  
            if j <= evaluation_dataset_num // 2:  
                evaluation_records.append(record) # reserve top 50% of evaluation_data
            else:  
                remaining_records.append(record)  
                
    if os.path.exists(evaluation_dataset_path):
        os.remove(evaluation_dataset_path)
        
    # save training dataset
    with h5py.File(evaluation_dataset_path, 'w') as e_file:
        for idx, record in enumerate(evaluation_records):
            group = e_file.create_group(f'record_{idx}')
            for key, data in record.items():
                group.create_dataset(key, data=data)
        
    # record all records
    if remaining_dataset_path is not None:
        with h5py.File(remaining_dataset_path, 'r') as file:
            for group in file.values():
                record = {}
                for name, data in group.items():
                    if data.shape == ():
                        record[name] = data[()]  
                    else:
                        record[name] = data[:]  
                remaining_records.append(record)
            
        if os.path.exists(remaining_dataset_path):
            os.remove(remaining_dataset_path)
                    
        with h5py.File(remaining_dataset_path, 'w') as r_file:
            for idx, record in enumerate(remaining_records):
                group = r_file.create_group(f'record_{idx}')
                for key, data in record.items():
                    group.create_dataset(key, data=data)


def expand_data(data, g):
    # get the third dim of data，data.shape: (batch, n_qubits, shots)
    h = data.size(2)
    
    # ensure h < g
    if h > g:
        raise ValueError("the dimension of data should be less than g")
    
    # expand the data to dim (d1, d2, g) by repeating data on the last dim
    expanded_bits = data.repeat(1, 1, g // h + 1)  
    
    # delete the redundant part if g//h != 0
    if expanded_bits.shape[2] > g:
        expanded_bits = expanded_bits[:, :, :g]
    
    return expanded_bits



import os
import shutil


def clear_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"error when deleting {file_path}: {e}")
    else:
        print(f"filefolder '{folder_path}' not exist or not valid")
