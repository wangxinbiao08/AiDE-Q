import numpy as np
from prob_cal import cal_entropy
from net.Qtype_datasetloader import DatasetLoader_Qtype
import matplotlib.pyplot as plt


dataset_path_list = ['../dataset/dataset_XXZ_50.h5']
differents = []
for idx, (n_qubits, bits, recipes, renyi_Entropy_3q) in enumerate(DatasetLoader_Qtype(dataset_path_list, 1)):
    recipes = recipes.swapaxes(1, 2).squeeze().numpy().astype(np.int8)[0:128]
    bits = bits.swapaxes(1, 2).squeeze().numpy().astype(np.int8)[0:128]
    bits = (bits == 1).astype(int)
    entropy_probing = cal_entropy(recipes, bits, [0, 1])
    differents.append(renyi_Entropy_3q[0][2] - entropy_probing)
    if idx % 100 == 0:
        print("finish", idx)

# 转换 differents 为 NumPy 数组
differents = np.array(differents)
print(differents.shape)
# 绘制直方图
plt.figure(figsize=(8, 6))
plt.hist(differents, bins=50, edgecolor='black', alpha=0.7)  # bins 控制区间数
plt.title("Distribution of Differences (Renyi_Entropy_3q - entropy_probing)")
plt.xlabel("Differences")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()





end = "end"