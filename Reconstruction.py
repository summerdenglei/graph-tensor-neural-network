
import LoadData as LD
import BuildModel as BM
import ReconstructionImage as RI
import DefineParam as DP
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 1. Input: Parameters
pixel_w, pixel_h, batchSize, nPhase, nTrainData, nValData, learningRate, nEpoch, nOfModel, ncpkt, trainFile, valFile, testFile, saveDir, modelDir = DP.get_param()


# 2. Data Loading
print('-------------------------------------\nLoading Data...\n-------------------------------------\n')
testLabel = LD.load_test_data(mat73=False)#True--False
trainPhi = np.ones([batchSize, 12, 24, 144])#改动：240，320改为120，160

L= testLabel.shape[3]
num_missing = int(0.9*L)
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

testPhi = trainPhi.astype('float32')


# 3. Model Building
print('-------------------------------------\nBuilding and Restoring Model...\n-------------------------------------\n')
sess, saver, Xinput, Xoutput, Yinput, Epoch_num, prediction, transField = BM.build_model(testPhi, missing_index, restore=True)


# 4. Image reconstruction
print('-------------------------------------\nReconstructing Image...\n-------------------------------------\n')
RI.reconstruct_image(sess, Yinput, Epoch_num, prediction, transField, Xinput, Xoutput, testLabel, testPhi, missing_index)
print('-------------------------------------\nReconstructing Accomplished.\n-------------------------------------\n')





