import csv
from collections import defaultdict

import numpy as np


def hamming_distance(s1, s2):

    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))


def cal_entropy(recipts, data, sub_sys):
    
    recipes_dict = defaultdict(int)
    shots = data.shape[0]
    for i in range(shots):
        numbers_recipt = [recipts[i,k] for k in sub_sys]
        binary_string_recipe = ''.join(str(num) for num in numbers_recipt)
        if binary_string_recipe not in recipes_dict:
            recipes_dict[binary_string_recipe] = [i]
        else:
            recipes_dict[binary_string_recipe].append(i)
    
    trace_rho_square = 0
    for recipe in list(recipes_dict.keys()):
        count_dict = defaultdict(int)
        for j in recipes_dict[recipe]:
            numbers = [data[j,k] for k in sub_sys]
            binary_string = ''.join(str(num) for num in numbers)
            count_dict[binary_string] += 1
        sum_x = 0
        recipt_shot = len(recipes_dict[recipe])
        for key1, value1 in count_dict.items():
            for key2, value2 in count_dict.items():
                distance = hamming_distance(key1, key2)
                sum_x += 2**(len(sub_sys)) * ((-2) ** (-distance)) * (value1/recipt_shot) * (value2/recipt_shot)
        trace_rho_square += sum_x
    entropy = -np.log2(trace_rho_square/len(recipes_dict))
    return entropy