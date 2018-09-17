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
# change str to float
data = data.apply(lambda x: pd.to_numeric(x))

for i in range(shape[0]):
    # print(i)
    # compute dissimilarity matrix for sex
    diss_matrix_sex[i:, i] = np.abs(data.iloc[i:,0] - data.iloc[i,0]) > 0
    # compute dissimilarity matrix for other attributes
    temp = pd.DataFrame(np.sum(np.abs(data.iloc[i:,1:] - data.iloc[i,1:].T), axis=1))
    if i != shape[0]-1:
        # normalized data with max and min
        diss_matrix[i:, i] = temp.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))).values.reshape(-1,)
    else:
        diss_matrix[i:,i] = 0

print(diss_matrix.shape, diss_matrix_sex.shape)
dissimilarity_matrix = (diss_matrix_sex + diss_matrix) / 2
dissimilarity_matrix_df = pd.DataFrame(dissimilarity_matrix)
# dissimilarity_matrix_df = dissimilarity_matrix_df.T + dissimilarity_matrix_df
dissimilarity_matrix_df.to_csv('dissimilarity_matrix_half.csv', header=None, index=False)
print(dissimilarity_matrix)
