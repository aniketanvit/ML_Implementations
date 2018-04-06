#### CREATE train_RBF.pkl and test_RBF.pkl from train_PCA.pkl###

import pickle
import numpy as np
import math
from scipy.linalg import inv

train_cluster=pickle.load(open( "Kmean_cluster.pkl", "rb" ), encoding="latin1")

train_data=pickle.load(open( "train_PCA.pkl", "rb" ),encoding ='latin1')
test_data=pickle.load(open( "test_PCA.pkl", "rb" ),encoding ='latin1')



train_feature = train_data['data']
train_target = train_data['target']

test_feature = test_data['data']
test_target = test_data['target']


num_features_old = train_feature.shape[1]
num_features_new = max(train_cluster)
num_data = train_feature.shape[0]

phi_means = np.zeros((num_features_new, num_features_old), dtype=float)
#phi_variances = np.zeros((num_features_new, num_features_old), dtype=float)

for i in range(0, num_features_new):
    temp_arr = np.take(train_feature, np.where(train_cluster == i)[0], axis=0)
    np.mean(temp_arr, axis=0, out=phi_means[i])
    # cov0 = np.cov(temp_arr, rowvar=False)
    # if temp_arr.shape[0] > 1:
    #     phi_variances[i] = (np.diag(cov0)) * np.eye(num_features_old, dtype=float)
    # else:
    #     phi_variances[i] = np.ones(num_features_old, dtype=float)

def new_feature(x, i):
    x_minus_mu = x - phi_means[i]
    new_val = np.exp(-0.1*(np.dot(x_minus_mu, x_minus_mu)))
    return new_val

def calc_new_features(x):
    temp = np.zeros(num_features_new, dtype=float)
    for i in range(num_features_new):
        temp[i] = new_feature(x, i)
    return temp


train_feature_new = np.apply_along_axis(calc_new_features, 1, train_feature)
test_data_new = np.apply_along_axis(calc_new_features, 1, test_feature)

new_train_dict = {}
new_test_dic = {}

new_train_dict["data"] = train_feature_new
new_train_dict["target"] = train_target

new_test_dic["data"] = test_data_new
new_test_dic["target"] = test_target

pickling_on = open("train_RBF.pkl", "wb+")
pickle.dump(new_train_dict, pickling_on)
pickling_on.close()

pickling_on1 = open("test_RBF.pkl", "wb+")
pickle.dump(new_test_dic, pickling_on1)
pickling_on1.close()

print("done...")