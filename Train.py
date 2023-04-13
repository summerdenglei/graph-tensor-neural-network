#       Graph-Tensor Neural Networks for Network Traffic Data Imputation
#


import LoadData as LD
import numpy as np
import BuildModel as BM
import TrainModel as TM
import DefineParam as DP
import os
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 1. Input: Parameters
pixel_w, pixel_h, batchSize, nPhase, nTrainData, nValData, learningRate, nEpoch, nOfModel, ncpkt, trainFile, valFile, saveDir, modelDir = DP.get_param()


# 2. Data Loading
print('-------------------------------------\nLoading Data...\n-------------------------------------\n')
trainLabel, valLabel = LD.load_train_data(mat73=False)#

trainPhi = np.ones([batchSize, 12, 24, 144])#

L= trainLabel.shape[3]

num_missing = round(0.7*L) # 0.7 is the missing rate
index = np.arange(L, dtype=int)
np.random.seed(1)
np.random.shuffle(index)
# sio.savemat("index.mat", {'index': index})
# idx = sio.loadmat('.\index_144.mat')
# index = idx['randindex1']
# index=index.astype(np.int32)

missing_index = (index[:num_missing])

print(missing_index)

for index_x in missing_index:
    trainPhi[:, :, :, index_x] = 0

trainPhi = trainPhi.astype('float32')


# 3. Model Building
print('-------------------------------------\nBuilding Model...\n-------------------------------------\n')
sess, saver, Xinput, Xoutput, Epoch_num, costMean, costSymmetric, costSparsity, optmAll, Yinput, prediction, lambdaStep, softThr, transField = BM.build_model(trainPhi, missing_index)


# 4. Model Training
print('-------------------------------------\nTraining Model...\n-------------------------------------\n')
TM.train_model(sess, saver, costMean, costSymmetric, costSparsity, optmAll, Yinput, prediction, trainLabel, valLabel, trainPhi, Xinput, Xoutput, Epoch_num, lambdaStep, softThr, missing_index, transField)
print('-------------------------------------\nTraining Accomplished.\n-------------------------------------\n')







