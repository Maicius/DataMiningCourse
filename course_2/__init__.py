import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

debug = False

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

def calculate_gain_split_for_discrect(X, y):
    X = X.reshape(-1,1)
    targets = pd.unique(y)
    info_d = calculate_expect_info(y)
    gain_list = []
    total_num = y.size
    if debug:
        print('info D:', info_d)
        print('begin to calculate info_i')
    for i in range(X.shape[1]):
        data_i = X[:,i]
        type_i = pd.unique(data_i)
        print(type_i)
        Info_i = 0
        for type in type_i:
            data_type = data_i[data_i == type]
            type_i_total = data_type.size
            if debug:
                print('type_i:', type_i_total, type)
            info_i = 0
            for target in targets:
                type_i_yes = y[y == target].size
                if debug:
                    print('target_num:', target, type_i_yes)
                info_i +=  - (type_i_yes / type_i_total) * np.log2((type_i_yes / type_i_total))
                if debug:
                    print('info_i:', info_i)
            Info_i += (type_i_total / total_num) * info_i
        gain_i = info_d - Info_i
        gain_list.append(gain_i)
    # print('gain_list:', gain_list)
    return gain_list


def calculate_expect_info(y):
    """
    计算期望信息
    :param y:
    :return:
    """
    targets = pd.unique(y)
    print(targets)
    info_d = 0
    total_num = y.size
    for target in targets:
        target_num = y[y == target].size
        if debug:
            print('target_num:', target, target_num)
        info_d += -(target_num / total_num) * np.log2((target_num / total_num))
    return info_d

def calculate_gain_split_for_continue(X, y):
    """

    :param X:
    :param y:
    :return: X中每个属性信息增益
    """

    # data = np.c_[X, y]
    expect_info_list = []
    for i in range(X.shape[1]):
        expect_info_list.append([])
        data = X[X[:, i].argsort()]
        for j in range(data.shape[0] - 1):
            split_line = (data[j, i] + data[j + 1, i]) / 2
            discrect_data_x = data[:, i] >= split_line
            expect_info = calculate_expect_info(discrect_data_x)
            print(expect_info, split_line)
            expect_info_list[i].append((expect_info, split_line))
    best_gain = []
    for i in range(len(expect_info_list)):
        best_split = max(expect_info_list[i], key= lambda x: x[0])[1]
        discrect_data_x = X[:, i] >= best_split
        gain = calculate_gain_split_for_discrect(discrect_data_x, y)[0]
        best_gain.append(gain)
    print(best_gain)
    return best_gain


print(X.shape, y.shape)
calculate_gain_split_for_continue(X, y)

