from net.Qtype_train_val import Qtype_train
from net.Qtype_dataengine import Qtype_dataengine
from utils.utils import split_datasets, clear_folder, mend_datasets, reallocate_datasets
import os
import h5py
import torch
import shutil


device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")  # ""cpu" or "cuda:0" where "0" is the specified index of used gpu
# pre-train hyper-params
n_qubits = 50
max_shots = 1024
max_epoch = 300
learning_rate = 0.0001 # 0.0001
batch_size = 64
start_early_stop_epoch_pt = 100
start_early_stop_epoch_de = 30

all_train_records_num = 3200
val_records_num = 160

p_noise = 0  # fixed, not need to change
# data-engine hyper-params
dataengine_repeat_time = 6
subset_num, subset_precentage, select_percentage = 5, 0.25, 0.1

physic_condition = True

for folder_num in ['1', '2', '3', '4', '5']: # ['1', '2', '3']:   
    
    # clear_folder('temp')
    folder_path = f'folder_{folder_num}'
    if not os.path.exists(f'logs_val/{folder_path}'):
        os.makedirs(f'logs_val/{folder_path}')
    if not os.path.exists(f'model_val/{folder_path}'):
        os.makedirs(f'model_val/{folder_path}')

    for dataengine_rate in [0.6, 0.8]:  #[0.2, 0.4, 0.6]:
        for dataengine_shots in [32, 512]:
            prefix = f'unlabel_rate_n{n_qubits}_r{dataengine_rate}_s{dataengine_shots}'
            dataset_path = f'dataset/dataset_XXZ_{n_qubits}.h5'
            if n_qubits == 50:
                path_source = os.path.abspath('..')
                dataset_path = f'{path_source}/XXZ_model_probing/{dataset_path}'

            if not os.path.exists(f'temp/{prefix}'):
                os.makedirs(f'temp/{prefix}')
            
            clear_folder(f'temp/{prefix}')
            
            train_dataset_path = f'temp/{prefix}/train_dataset.h5'
            val_dataset_path = f'temp/{prefix}/val_dataset.h5'

            selected_dataset_path = f'temp/{prefix}/selected_dataset.h5'
            evaluation_dataset_path = f'temp/{prefix}/evaluation_dataset.h5'
            remaining_dataset_path = f'temp/{prefix}/remaining_dataset.h5'

            dataengine_logs_path = f'logs_val/{folder_path}/dataengine_logs_{prefix}.txt'

            (train_records_num, dataengine_records_num) = (int(all_train_records_num*(1-dataengine_rate)), int(all_train_records_num*dataengine_rate))
            if not os.path.exists(train_dataset_path) or not os.path.exists(val_dataset_path):
                split_datasets(dataset_path, train_dataset_path, val_dataset_path, remaining_dataset_path,
                               train_records_num, val_records_num, dataengine_records_num, p_noise, max_shots)
            pretrain_model_save_path = f'model_val/{folder_path}/{prefix}_pretrain_model.pt'
            logs_save_path = f'logs_val/{folder_path}/{prefix}_logs_test.txt'
            
            if os.path.exists(dataengine_logs_path):
                os.remove(dataengine_logs_path)
            if os.path.exists(logs_save_path):
                os.remove(logs_save_path)

            # pre-train
            PRE_mse, PRE_R2 = Qtype_train([train_dataset_path], pretrain_model_save_path, logs_save_path,
                                          max_epoch, learning_rate, batch_size, start_early_stop_epoch_pt,
                                          device, val_dataset_path,
                                          val_shots = dataengine_shots,
                                          physic_condition = physic_condition,
                                          n_qubits=n_qubits,
                                          num_measurements=6,
                                          shots=int(max_shots),
                                          emb_dim=128,
                                          num_heads=4,
                                          num_layers=2,
                                          dim_feedforward=128)

            with open(dataengine_logs_path, 'a') as file:
                file.write(f'initial R2:{PRE_R2}\n')

            last_mse = PRE_mse
            last_R2 = PRE_R2
            i = 0
            
            
            # start dataengin
            while i < dataengine_repeat_time:
                if os.path.exists(remaining_dataset_path):
                    with h5py.File(remaining_dataset_path, 'r') as f:
                        num_records_selected = len(f.keys())
                        if num_records_selected < int(all_train_records_num*dataengine_rate*0.125):
                            break

                # dataengine for data selection
                data_engine_model_save_path = f'model_val/{folder_path}/{prefix}_DE_model_{i}.pt'
                repeat_time = int(max_shots/dataengine_shots)
                
                # model evaluation setting
                if i == 0:
                    retrain_dataset_path_list = [train_dataset_path, evaluation_dataset_path]
                    used_model_path = pretrain_model_save_path
                else:
                    retrain_dataset_path_list = [train_dataset_path, selected_dataset_path, evaluation_dataset_path]
                    used_model_path =  f'model_val/{folder_path}/{prefix}_DE_model_{i-1}.pt'
                    
                if i >= 2 and os.path.exists(f'model_val/{folder_path}/{prefix}_DE_model_{i-2}.pt'):
                    os.remove(f'model_val/{folder_path}/{prefix}_DE_model_{i-2}.pt')
                    
                Qtype_dataengine(evaluation_dataset_path, remaining_dataset_path, logs_save_path,
                                 batch_size, dataengine_shots, physic_condition,
                                 subset_num, subset_precentage, select_percentage,
                                 used_model_path, device,  # ? the used model_path should be iterated as data-engine runs
                                 n_qubits=n_qubits,
                                 num_measurements=6,
                                 shots=int(max_shots),
                                 emb_dim=128,
                                 num_heads=4,
                                 num_layers=2,
                                 dim_feedforward=128
                                 )

                # -------------------------------------------------------------------
                select_num = 0
                evaluation_num = 0
                remain_num = 0
                if os.path.exists(selected_dataset_path):
                    with h5py.File(selected_dataset_path, 'r') as f:
                        select_num = len(f.keys())
                if os.path.exists(evaluation_dataset_path):
                    with h5py.File(evaluation_dataset_path, 'r') as f:
                        evaluation_num = len(f.keys())
                if os.path.exists(remaining_dataset_path):
                    with h5py.File(remaining_dataset_path, 'r') as f:
                        remain_num = len(f.keys())

                with open(dataengine_logs_path, 'a') as file:
                    file.write(f'current iteration: {i}\n')
                with open(dataengine_logs_path, 'a') as file:
                    file.write(f'selected data size: {select_num}, evaluation data size: {evaluation_num}, remain data size:{remain_num}\n')
                # -------------------------------------------------------------------

                # dataengine retrain
                print(f'Round DataEngine: {i}\n')
                # data setting
                if i == 0:
                    retrain_dataset_path_list = [train_dataset_path, evaluation_dataset_path]
                else:
                    retrain_dataset_path_list = [train_dataset_path, selected_dataset_path, evaluation_dataset_path]
                DE_mse, DE_R2 = Qtype_train(retrain_dataset_path_list, data_engine_model_save_path, logs_save_path,
                                            max_epoch, learning_rate, batch_size, start_early_stop_epoch_de,
                                            device, val_dataset_path, used_model_path, 
                                            val_shots = dataengine_shots,
                                            physic_condition = physic_condition,
                                            n_qubits=n_qubits,
                                            num_measurements=6,
                                            shots=int(max_shots),
                                            emb_dim=128,
                                            num_heads=4,
                                            num_layers=2,
                                            dim_feedforward=128)
                
                # avoid model performance decreasing
                count_back = 0
                while DE_R2 < last_R2:
                    if os.path.exists(evaluation_dataset_path):
                        with h5py.File(evaluation_dataset_path, 'r') as f:
                            num_records_evaluation = len(f.keys())
                            if num_records_evaluation <= 3 or count_back>=2:
                                break
                            
                    count_back += 1
                    with open(dataengine_logs_path, 'a') as file:
                        file.write(f'last R2:{last_R2}, new R2: {DE_R2}, model backward\n')
                    print(f'last R2:{last_R2}, new R2: {DE_R2}, model backward\n')
                    print(f'Round DataEngine Backward: {i}-{count_back}\n')
                    reallocate_datasets(evaluation_dataset_path) 
                    
                    if i == 0:
                        retrain_dataset_path_list = [train_dataset_path, evaluation_dataset_path]
                        used_model_path = pretrain_model_save_path
                    else:
                        retrain_dataset_path_list = [train_dataset_path, selected_dataset_path, evaluation_dataset_path]
                        used_model_path =  f'model_val/{folder_path}/{prefix}_DE_model_{i-1}.pt'
                    
                    DE_mse, DE_R2 = Qtype_train(retrain_dataset_path_list, data_engine_model_save_path, logs_save_path,
                                                max_epoch, learning_rate, batch_size, start_early_stop_epoch_de,
                                                device, val_dataset_path, used_model_path,
                                                val_shots = dataengine_shots,
                                                physic_condition = physic_condition,
                                                n_qubits=n_qubits,
                                                num_measurements=6,
                                                shots=int(max_shots),
                                                emb_dim=128,
                                                num_heads=4,
                                                num_layers=2,
                                                dim_feedforward=128)
                    
                if DE_R2 >= last_R2:
                    print(f'last R2:{last_R2}, new R2: {DE_R2}, model forward\n')
                    with open(dataengine_logs_path, 'a') as file:
                        file.write(f'last R2:{last_R2}, new R2: {DE_R2}, model forward\n')
                    if i == 0:
                        shutil.copy2(evaluation_dataset_path, selected_dataset_path)
                    else:
                        mend_datasets([selected_dataset_path, evaluation_dataset_path], selected_dataset_path)
                    last_mse = DE_mse
                    last_R2 = DE_R2
                    i = i + 1




