import h5py
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class Dataset_Qtype(Dataset):
    def __init__(self, dataset_path, physic_condition):
        self.dataset_path = dataset_path
        self.data = []
        with h5py.File(self.dataset_path, 'r') as f:
            for group_name in f:
                group = f[group_name]
                n_qubit = group['n_qubit'][()]
                bits = group['bits'][:]
                bits = (bits == 1).astype(int)  # 把-1映射到0，1映射到1
                recipes = group['recipes'][:]
                renyi_Entropy_3q = group['renyi_Entropy_3q'][:]
                if physic_condition == True:
                    param_J = group['param_J'][()]
                    param_g = group['param_g'][()]
                    self.data.append((n_qubit, bits, recipes, renyi_Entropy_3q, param_J, param_g))      
                else:
                    self.data.append((n_qubit, bits, recipes, renyi_Entropy_3q))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DatasetLoader_Qtype:
    def __init__(self, dataset_paths, batch_size, physic_condition=True):
        datasets = [Dataset_Qtype(dataset_path, physic_condition) for dataset_path in dataset_paths]
        combined_dataset = ConcatDataset(datasets)
        self.data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    def __iter__(self):
        return iter(self.data_loader)
