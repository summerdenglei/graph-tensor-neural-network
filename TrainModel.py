#       Video Synthesis via Transform-Based Tensor Neural Network
#                             Yimeng Zhang
#                               8/4/2020
#                         yz3397@columbia.edu


import numpy as np
import DefineParam as DP
import os
import math
from time import time
import scipy.io as sio

# Input: Parameters
pixel_w, pixel_h, batchSize, nPhase, nTrainData, nValData, learningRate, nEpoch, nOfModel, ncpkt, trainFile, valFile, testFile, saveDir, modelDir = DP.get_param()

outputFile = "log_int.txt"





# Model Training
def train_model(sess, saver, costMean, costSymmetric, costSparsity, optmAll, Yinput, prediction, trainLabel, valLabel, trainPhi, Xinput, Xoutput, Epoch_num, lambdaStep, softThr, missing_index, transField):

    iCostMeanLast = 1
    iCostSymmetricLast = 1
    iCostSparsityLast = 1
    iCostMeanChangeRate = 0
    iCostSymmetricChangeRate = 0
    iCostSParsityChangeRate = 0

    if not os.path.exists(modelDir):                                                     # Save Model
        os.makedirs(modelDir)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    out = open(outputFile, 'a')
    out.write('\n\n------------------------------------------------------------------------------------------------------------------------------------------')
    out.close()

    trainPart = np.random.permutation(nTrainData // batchSize)                           # Random Disorder
    valPart = np.random.permutation(nValData // batchSize)
    rse_epoch=[]

    for epoch in range(1, nEpoch + 1):
        epoch_num = epoch                                                                # Training
        batchCount = -1
        print("training epoch:%d/%d" % (epoch, nEpoch))
        for batchi in trainPart:                       
            batchCount += 1
            # print("training epoch:%d/%d batch:%d/%d, establishing dictionary" % (epoch, nEpoch, batchCount, len(trainPart)))
            xoutput = trainLabel[batchSize*batchi: batchSize*(batchi + 1), :, :, :]
            yinput = np.multiply(xoutput, trainPhi)
            xinput = np.multiply(xoutput, trainPhi)
           
            feedDict = {Xinput: xinput, Xoutput: xoutput, Yinput: yinput, Epoch_num: epoch_num}
            # print("training epoch:%d/%d batch:%d/%d, optmizing loss function" % (epoch, nEpoch, batchCount, len(trainPart)))
            sess.run(optmAll, feed_dict=feedDict)

        batchCount = -1                                                                  # Validation
        allInitPSNR = 0
        allPSNR = 0
        averse=[]
        aveMAE=[]
        aveRMSE = []
        aveR2 = []
        for batchi in valPart:
            batchCount += 1
            # print("validating epoch:%d/%d batch:%d/%d, establishing dictionary" % (epoch, nEpoch, batchCount, len(valPart)))
            xoutput = valLabel[batchSize*batchi: batchSize*(batchi + 1), :, :, :]
            yinput = np.multiply(xoutput, trainPhi)
            xinput = np.multiply(xoutput, trainPhi)

            initPSNR=0
            for index_x in missing_index:
                initPSNR += psnr(xinput[:, :, :, index_x], xoutput[:, :, :, index_x])
            initPSNR /= len(missing_index)
            # print("validating epoch:%d/%d batch:%d/%d, init PSNR: %.4f" % (epoch, nEpoch, batchCount, len(valPart), initPSNR))
            allInitPSNR += initPSNR            

            feedDict = {Xinput: xinput, Xoutput: xoutput, Yinput: yinput, Epoch_num: epoch_num}
            start = time()
            result = sess.run(prediction[-1], feed_dict=feedDict)
            end = time()

            result=np.abs(result)
            # if epoch==490:
            #     sio.savemat("./result/res%d.mat"%(batchi+1), {'res': result})

            recPSNR = 0

            for index_x in missing_index:
                recPSNR += psnr(result[:, :, :, index_x], xoutput[:, :, :, index_x])
            recPSNR /= len(missing_index)
            rse, rmse = Myrse(result, xoutput, missing_index)
            mae, R2 = MyMAE(result, xoutput, missing_index)
            # print("validating epoch:%d/%d batch:%d/%d, PSNR: %.4f, time: %.2f" % (epoch, nEpoch, batchCount, len(valPart), recPSNR, end-start))
            # print("validating epoch:%d/%d batch:%d/%d, RSE: %.4f, time: %.2f" % (epoch, nEpoch, batchCount, len(valPart), rse, end - start))
            allPSNR += recPSNR
        averse.append(rse)
        aveRMSE.append(rmse)
        aveMAE.append(mae)
        aveR2.append(R2)

        print("average RSE: %.4f, average MAE: %.4f, average RMSE: %.4f, average R2: %.4f,  time: %.6f" % (np.mean(averse), np.mean(aveMAE),np.mean(aveRMSE),np.mean(aveR2),  end - start))
        avgInitPSNR = allInitPSNR/np.maximum(len(valPart), 1) 
        avgPSNR = allPSNR/np.maximum(len(valPart), 1)
        validateInfo = "Avg init PSNR :%.4f, avg validating PSNR: %.4f\n" % (avgInitPSNR, avgPSNR)
        # print(validateInfo)
        rse_epoch.append(np.mean(averse))

        print("epoch:%02d/%02d calculating prediction loss" % (epoch, nEpoch))           # Loss Calculations
        iCostMean = sess.run(costMean, feed_dict=feedDict)
        iCostSymmetric = sess.run(costSymmetric, feed_dict=feedDict)
        iCostSparsity = sess.run(costSparsity, feed_dict=feedDict)

        if epoch == 0:
            iCostMeanChangeRate = 0
            iCostSymmetricChangeRate = 0
            iCostSparsityChangeRate = 0
        else:
            iCostMeanChangeRate = 100*(iCostMeanLast - iCostMean)/iCostMeanLast
            iCostSymmetricChangeRate = 100*(iCostSymmetricLast - iCostSymmetric)/iCostSymmetricLast
            iCostSparsityChangeRate = 100*(iCostSparsityLast - iCostSparsity)/iCostSparsityLast

        iCostMeanLast = iCostMean
        iCostSymmetricLast = iCostSymmetric
        iCostSparsityLast = iCostSparsity
        outputData = "\n[epoch:%02d/%02d]\tcostMean: %.5f(%.2f%%)\tcostSymmetric: %.5f(%.2f%%)\tcostSparsity: %.5f(%.2f%%)\n" % (epoch, nEpoch, iCostMean, iCostMeanChangeRate, iCostSymmetric, iCostSymmetricChangeRate, iCostSparsity, iCostSparsityChangeRate)
        print(outputData)
        lamb = sess.run(lambdaStep, feed_dict=feedDict)
        soft = sess.run(softThr, feed_dict=feedDict)
        varInfo = "[lambdaStep: %.2f, softThr: %.2f]\n" % (lamb, soft)
        print(varInfo)

        out = open(outputFile, 'a')
        out.write(outputData)
        out.write(varInfo)
        out.write(validateInfo)
        out.close()

        if epoch < 50 or epoch % 10 == 0:
            saver.save(sess, '%s/%d.cpkt' % (modelDir, epoch))
            # print('model saved\n')
    sio.savemat("2conv2d_rse_epoch03v2.mat", {'rse_epoch': rse_epoch})
    sess.close()










def Myrse(tensor11, tensor22, missing_index):
    tensor1=tensor11[:,:,:, missing_index]
    tensor2 = tensor22[:, :, :, missing_index]
    tensor1.astype(np.float32)
    tensor2.astype(np.float32)
    n1, n2, n3, n4 = tensor2.shape
    num = n1 * n2 * n3 * n4

    mse = np.sqrt(np.sum((tensor1 - tensor2)**2))
    mse2=np.sqrt( np.sum(tensor22**2))
    rse=mse / mse2
    rmse=rse/np.sqrt(num)
    return rse, rmse

def MyMAE(tensor11, tensor22, missing_index):
    tensor1 = tensor11[:, :, :, missing_index]
    tensor2 = tensor22[:, :, :, missing_index]
    tensor1.astype(np.float32)
    tensor2.astype(np.float32)
    D=(tensor1 - tensor2)
    n1, n2,n3,n4=D.shape
    num=n1*n2*n3*n4

    sumerror=np.sum(np.abs(D))
    MAE=sumerror/num

    avey=np.mean(tensor2)
    R2=1-((np.sum((tensor1 - tensor2)**2))/(np.sum((avey - tensor2)**2)))
    return MAE, R2

def init(Q, meas):
    meas = np.reshape(meas, (-1, 1024, 1), 'F')
    init = np.dot(Q, meas)
    init = np.reshape(init, (-1, 32, 32, 8), 'F')
    return init
