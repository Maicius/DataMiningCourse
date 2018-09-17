## This is code is for Abalone dataset
import numpy as np
import pandas as pd

## First load data, data name: Sex,Length,Diameter,Height,Whole weight,Shucked weight,Viscera weight,Shell weight,Rings
## Different from the iris data set, you should download the abalone data and put it and this code file into the same directory
dataMatrix = np.loadtxt("course_1/abalone.data", delimiter=",", dtype=str)
print("Here is data matrix : ")
print("##########################################################")
print(dataMatrix)
## Make dissimilarity matrix using Proximity Measures for sex, and Normalized Manhattand distance for other attributes.
# THis is your exercise !!

data = pd.DataFrame(dataMatrix)
i = 0

sex_dict = {
    'M': 1,
    'I': 2,
    'F': 3
}
data[0] = data[0].map(sex_dict)
shape = data.shape
# normalized data with max and min
normalized_data = data.apply(
    lambda x: (pd.to_numeric(x) - np.min(pd.to_numeric(x))) / (np.max(pd.to_numeric(x)) - np.min(pd.to_numeric(x))))
print(normalized_data)
diss_matrix_sex = np.zeros((shape[0], shape[0]))
diss_matrix = np.zeros((shape[0], shape[0]))
### do it in for loop, very very slow

# for i in range(shape[0]):
#     for j in range(i, shape[0]):
#         print(i, j)
#         diss_martix[i, j] = np.sum(np.abs(normalized_data.iloc[i, :] - normalized_data.iloc[j, :]))
# diss_data = pd.DataFrame(diss_martix)
# diss_data = diss_data.T + diss_data

### do it with broadcast, very quick
for i in range(shape[0]):
    print(i)
    # compute disssimilary matrix for sex
    diss_matrix_sex[:, i] = (data[0] - data.iloc[i,0]) > 0
    # compute disssimilary matrix for other attributes
    diss_matrix[:, i] = np.sum(np.abs(normalized_data.iloc[:,1:] - normalized_data.iloc[i,1:].T), axis=1)
print(diss_matrix.shape, diss_matrix_sex.shape)
diss_matrix = (diss_matrix_sex + diss_matrix) / 2
diss_data = pd.DataFrame(diss_matrix)
diss_data.to_csv('dissimilarity_matrix.csv', header=None, index=False)
print(diss_matrix)
