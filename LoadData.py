#       Video Synthesis via Transform-Based Tensor Neural Network
#                             Yimeng Zhang
#                               8/4/2020
#                         yz3397@columbia.edu


import tensorflow as tf
import scipy.io as sio
import numpy as np
import DefineParam as DP
import h5py


# Get param
pixel_w, pixel_h, batchSize, nPhase, nTrainData, nValData, learningRate, nEpoch, nOfModel, ncpkt, trainFile,  valFile, saveDir, modelDir = DP.get_param()






# Training Data Loading
def load_train_data(mat73=False):
    if mat73 == True:
        trainData = h5py.File(trainFile)
        trainLabel = np.transpose(trainData['sub_data'], [3, 2, 1, 0])
    else:
        trainData = sio.loadmat(trainFile)
        trainLabel = trainData['sub_data']


    if mat73 == True:
        valData = h5py.File(valFile)
        valLabel = np.transpose(valData['sub_data'], [3, 2, 1, 0])
    else:
        valData = sio.loadmat(valFile)
        valLabel = valData['sub_data']

    print("nOfModel: %d" % nOfModel)
    print(np.shape(trainLabel))

    del trainData
    del valData
    return trainLabel, valLabel







# Testing Data Loading
def load_test_data(mat73=False):
    if mat73 == True:
        testData = h5py.File(testFile)
        testLabel = np.transpose(testData['sub_data'], [3, 2, 1, 0])
    else:
        testData = sio.loadmat(testFile)
        testLabel = testData['sub_data']              # labels

    print(np.shape(testLabel))

    del testData
    return testLabel








# Essential Computations
def pre_calculate(phi):
    Xinput = tf.placeholder(tf.float32, [None, pixel_h, pixel_w, nOfModel])                  # After Init
    Xoutput = tf.placeholder(tf.float32, [None, pixel_h, pixel_w, nOfModel])
    Yinput = tf.placeholder(tf.float32, [None, pixel_h, pixel_w, nOfModel])                  # After sampling
    Epoch_num = tf.placeholder(tf.float32)

    Z1 = np.zeros(phi.shape)
    Y1 = np.zeros(phi.shape)
    Z1=Z1.astype('float32')
    Y1 = Y1.astype('float32')

    Z=tf.constant(Z1)
    Y=tf.constant(Y1)

    phit=1-phi
    Phi = tf.constant(phi)
    PhiT = tf.constant(phit)

    return Xinput, Xoutput, Phi, PhiT, Yinput, Epoch_num, Y, Z













