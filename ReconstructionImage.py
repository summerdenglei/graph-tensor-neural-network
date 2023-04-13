#       Video Synthesis via Transform-Based Tensor Neural Network
#                             Yimeng Zhang
#                               8/4/2020
#                         yz3397@columbia.edu

import scipy.io as sio
import numpy as np
from time import time
import math
import DefineParam as DP
import os

# Input: Parameters
pixel_w, pixel_h, batchSize, nPhase, nTrainData, nValData, learningRate, nEpoch, nOfModel, ncpkt, testFile, valFile, testFile, saveDir, modelDir = DP.get_param()





# Testing
def reconstruct_image(sess, Yinput, Epoch_num, prediction, transField, Xinput, Xoutput, testLabel, testPhi, missing_index):
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    avgInitPSNR = 0
    avgRecPSNR = 0
    epoch_num = 300
    averse=[]
    nTestData = 1
    testPart = np.random.permutation(nTestData // batchSize)
    batchCount = -1
    allInitPSNR = 0
    allRecPSNR = 0
    for batchi in testPart:
        batchCount += 1
        print("batch:%d/%d, establishing dictionary" % ( batchCount, len(testPart)))
        xoutput = testLabel[batchSize*batchi: batchSize*(batchi + 1), :, :, :]
        yinput = np.multiply(xoutput, testPhi)
        xinput = np.multiply(xoutput, testPhi)

        initPSNR=0
        for index_x in missing_index:
            initPSNR += psnr(xinput[:, :, :, index_x], xoutput[:, :, :, index_x])
        initPSNR /= len(missing_index)
        print(" batch:%d/%d, init PSNR: %.4f" % ( batchCount, len(testPart), initPSNR))
        allInitPSNR += initPSNR            

        feedDict = {Xinput: xinput, Xoutput: xoutput, Yinput: yinput, Epoch_num: epoch_num}
        start = time()
        result = sess.run(prediction[-1], feed_dict = feedDict)            
        end = time()

        sio.savemat("res.mat", {'res': result})

        recPSNR = 0
        rse=0
        for index_x in missing_index:
            recPSNR += psnr(result[:, :, :, index_x], xoutput[:, :, :, index_x])

        recPSNR /= len(missing_index)
        rse = Myrse(result[:, :, :, missing_index], xoutput[:, :, :, missing_index])
        print(" batch:%d/%d, PSNR: %.4f, time: %.2f" % (batchCount, len(testPart), recPSNR, end-start))
        print(" batch:%d/%d, RSE: %.4f, time: %.2f" % (batchCount, len(testPart), rse, end - start))
        allRecPSNR += recPSNR
    averse.append(rse)
    print("average RSE: %.4f, time: %.2f" % (np.mean(averse), end - start))
    allAvgInitPSNR = allInitPSNR/np.maximum(len(testPart), 1)
    allAvgRecPSNR = allRecPSNR/np.maximum(len(testPart), 1)

    print("All avg init PSNR:%.2f" % allAvgInitPSNR)
    print("All avg rec PSNR:%.2f" % allAvgRecPSNR)

    sess.close()






# PSNR Calculation
def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    return 20*math.log10(1.0/math.sqrt(mse))

def Myrse(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.sqrt(np.sum((img1 - img2)**2))
    mse2=np.sqrt( np.sum(img2**2))
    return mse/mse2