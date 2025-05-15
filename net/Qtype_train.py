import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from net.Qtype_net import Qtype
from net.Qtype_datasetloader import Dataset_Qtype, DatasetLoader_Qtype
import os

def R2_score(preds, labels):
    # 计算均方误差
    ss_res = np.sum((labels - preds) ** 2)  # Residual sum of squares
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)  # Total sum of squares

    # 计算R^2
    r2 = 1 - (ss_res / ss_tot)
    return r2


def Qtype_train(dataset_path_list, model_save_path, logs_save_path,
                num_epochs, learning_rate, batch_size, start_early_stop_epoch,
                device="cuda:0", val_dataset_path=None, pretrain_model_path=None,
                **param):
    logs = ''
    shots = param.get('shots', None)
    Net = Qtype(**param).to(device)
    if pretrain_model_path is not None:
        Net.load_state_dict(torch.load(pretrain_model_path, map_location=device))

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(Net.parameters(), lr=learning_rate, weight_decay=1e-3)

    model_parameters_history = []
    train_mse_history = []
    val_mse_history = []
    R2_history = []

    patience = int(15)  # 最大容忍验证集指标未改进的 epoch 数
    best_val_mse = float('inf')  # 保存最佳验证集 mse
    best_epoch = 0  # 保存最佳 epoch
    no_improvement_count = 0  # 未改进次数计数
    start_early_stop_epoch = start_early_stop_epoch  # 设置早停开始的最小 epoch 数

    top_val_mse = []
    for epoch in range(num_epochs):
        train_mse = 0
        train_records = 0
        for idx, (n_qubits, bits, recipes, renyi_Entropy_3q) in enumerate(DatasetLoader_Qtype(dataset_path_list, batch_size)):

            # train
            Net.train()
            input_data = (bits + recipes * 2).to(device).float()
            input_data = input_data[:, :, 0:shots]
            input_data = input_data.swapaxes(1, 2)
            output, orthogonal_loss = Net(input_data)
            target_tensor = renyi_Entropy_3q.clone().detach().to(dtype=output.dtype, device=device)
            if torch.isnan(target_tensor).any():
                print("there is a nan in target_tensor")
            mse_loss = criterion(output, target_tensor)

            train_mse += mse_loss.item()
            train_records += 1
            loss = mse_loss + orthogonal_loss * 0.025
            optimizer.zero_grad() # 清空梯度
            loss.backward()   # 反向传播计算梯度
            optimizer.step()   # 更新参数
        avg_train_mse = train_mse / train_records
        train_mse_history.append(avg_train_mse)
        model_parameters_history.append(Net.state_dict())

        if val_dataset_path is not None:
            Net.eval()
            val_mse = 0
            val_records = 0
            preds = []
            labels = []
            with torch.no_grad():
                for idx, (n_qubits, bits, recipes, renyi_Entropy_3q) in enumerate(DatasetLoader_Qtype([val_dataset_path], batch_size)):
                    n_qubits, bits, recipes, renyi_Entropy_3q = n_qubits.to(device), bits.to(device), recipes.to(
                        device), renyi_Entropy_3q.to(device)
                    input_data = (bits + recipes * 2).to(device).float()
                    input_data = input_data[:, :, 0:shots]

                    input_data = input_data.swapaxes(1, 2)
                    output, orthogonal_loss = Net(input_data)
                    mse_loss = criterion(output, renyi_Entropy_3q)
                    val_mse += mse_loss.item()
                    val_records += 1

                    preds.append(output.detach().view(-1).cpu().numpy())  # 展平并转为 NumPy 数组
                    labels.append(renyi_Entropy_3q.view(-1).cpu().numpy())

            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            R2 = R2_score(preds, labels)

            avg_val_mse = val_mse / val_records
            val_mse_history.append(avg_val_mse)
            R2_history.append(R2)
            if len(top_val_mse) < 5:
                top_val_mse.append((avg_val_mse, epoch, Net.state_dict()))
                top_val_mse.sort(key=lambda x: x[0])  # 按照 mse 排序
            elif avg_val_mse < top_val_mse[-1][0]:
                top_val_mse[-1] = (avg_val_mse, epoch, Net.state_dict())
                top_val_mse.sort(key=lambda x: x[0])  # 重新排序

            if avg_val_mse < best_val_mse - 1e-4:  # 1e-4 是 delta
                best_val_mse = avg_val_mse
                best_epoch = epoch
                no_improvement_count = 0  # 重置计数
                print(f'epoch:{epoch+1},  train_mse:{avg_train_mse}, val_mse:{avg_val_mse}, R2:{R2}, best model update')
                with open(logs_save_path, 'a') as file:
                    file.write(f'epoch:{epoch+1},  train_mse:{avg_train_mse}, val_mse:{avg_val_mse}, R2:{R2}, best model update\n')
            else:
                no_improvement_count += 1
                print(f'epoch:{epoch+1},  train_mse:{avg_train_mse}, val_mse:{avg_val_mse}, R2:{R2}, no improvement')
                with open(logs_save_path, 'a') as file:
                    file.write(f'epoch:{epoch+1},  train_mse:{avg_train_mse}, val_mse:{avg_val_mse}, R2:{R2}, no improvement\n')

            # 检查早停条件
            if epoch >= start_early_stop_epoch and no_improvement_count >= patience:
                logs = f'Early stopping at epoch {epoch + 1}. Best train_mse: {train_mse_history[best_epoch]:.6f}, val_mse: {val_mse_history[best_epoch]:.6f}, R2:{R2_history[best_epoch]}, at epoch {best_epoch + 1}.'
                print(logs)
                with open(logs_save_path, 'a') as file:
                    file.write(logs)
                break
        else:
            print(f'epoch:{epoch+1},  train_mse:{avg_train_mse}')
            logs = f'last train_mse:{train_mse_history[-1]} in epoch:{epoch + 1}'

    top_val_mse_values = [mse[0] for mse in top_val_mse]
    top_val_mse_sorted = sorted(top_val_mse_values)
    top_val_mse_avg = top_val_mse_sorted[0]
    # top_3_val_mse = top_val_mse_sorted[2:]  # 去掉最小的两个
    # top_val_mse_avg = np.mean(top_3_val_mse)

    R2_history_sorted = sorted(R2_history, reverse=True)
    top_R2_avg = R2_history_sorted[0]
    # top_3_to_5 = R2_history_sorted[2:5]
    # top_R2_avg = sum(top_3_to_5) / len(top_3_to_5)

    print(f'Top 3 val_mse average: {top_val_mse_avg:.6f}, R2 average:{top_R2_avg:.6f}')
    logs = f'\nTop 3 val_mse average: {top_val_mse_avg:.6f}, R2 average:{top_R2_avg:.6f}\n'

    
    if os.path.exists(model_save_path):
        os.remove(model_save_path)
        
    # 保存最佳模型
    torch.save(top_val_mse[0][2], model_save_path)  # 保存第一个（最优）的模型
    with open(logs_save_path, 'a') as file:
        file.write(logs)
    return top_val_mse_avg, top_R2_avg


