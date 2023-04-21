#       Video Synthesis via Transform-Based Tensor Neural Network
#                             Yimeng Zhang
#                               8/4/2020
#                         yz3397@columbia.edu

# Parameter Definitions
def get_param():    
    pixel_w = 24
    pixel_h = 12
    batchSize = 1
    nPhase = 4
    nTrainData = 140#2808--160
    nValData = 28#1396--20
    learningRate = 0.00001
    nEpoch = 1000#300--2
    nOfModel = 144
    ncpkt = 290 #290--2

    trainFile = './trainData/graph_traffic_train.mat' #training data
    #trainFile = 'E:/DengLei/Transform-Based-Tensor-Net-master\trainData/trainUCF.mat'
    valFile = './valData/graph_traffic_val.mat'  #test dataset. Validation dataset is also the test dataset.

    saveDir = './recImg/recImgFinal'
    modelDir = './Model_Int'

    return pixel_w, pixel_h, batchSize, nPhase,nTrainData, nValData, learningRate, nEpoch, nOfModel, ncpkt, trainFile, valFile, saveDir, modelDir
