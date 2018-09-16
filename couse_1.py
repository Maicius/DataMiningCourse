## This is code is for Abalone dataset
import numpy as np
import pandas as pd

## First load data, data name: Sex,Length,Diameter,Height,Whole weight,Shucked weight,Viscera weight,Shell weight,Rings
## Different from the iris data set, you should download the abalone data and put it and this code file into the same directory
dataMatrix = np.loadtxt("abalone.data.txt", delimiter=",", dtype=str)
print("Here is data matrix : ")
print("##########################################################")
print(dataMatrix)
data = pd.DataFrame(dataMatrix)
sex_dict = {
    'M': 1,
    'F': 2,
    'I': 3
}
data[0] = data[0].map(sex_dict)
shape = data.shape
# normalized data with max and min
normalized_data = data.apply(
    lambda x: (pd.to_numeric(x) - np.min(pd.to_numeric(x))) / (np.max(pd.to_numeric(x)) - np.min(pd.to_numeric(x))))
print(normalized_data)
diss_martix = np.zeros((shape[0], shape[0]))
for i in range(shape[0]):
    for j in range(i, shape[0]):
        print(i, j)
        diss_martix[i, j] = np.sum(np.abs(normalized_data.iloc[i, :] - normalized_data.iloc[j, :]))
## Make dissimilarity matrix using Proximity Measures for sex, and Normalized Manhattand distance for other attributes.
# THis is your exercise !!
diss_data = pd.DataFrame(diss_martix)
diss_data = diss_data.T + diss_data
diss_data.to_csv('dissimilarity_data.csv', header=None, index=False)
print(diss_martix)
