#       Video Synthesis via Transform-Based Tensor Neural Network
#                             Yimeng Zhang
#                               8/4/2020
#                         yz3397@columbia.edu

import tensorflow as tf
import LoadData as LD
import DefineParam as DP


# Input: Parameters
pixel_w, pixel_h, batchSize, nPhase, nTrainData, nValData, learningRate, nEpoch, nOfModel, ncpkt, trainFile, valFile, testFile, saveDir, modelDir = DP.get_param()





# Model Building
def build_model(phi, missing_index, restore=False):

    Xinput, Xoutput, Phi, PhiT, Yinput, Epoch_num, Y, Z = LD.pre_calculate(phi)#占位

    prediction, predictionSymmetric, transField, lambdaStep, softThr = inference_ista(Y, Z, Xinput, Xoutput, Phi, PhiT, Yinput, Epoch_num, reuse=False)

    costMean, costSymmetric = compute_cost(prediction, predictionSymmetric, Xoutput, phi)

    if restore is False:
        costSparsity = compute_sparsity(transField)
        costAll =costMean + 0.01*costSymmetric
        # + 0.001 * costSparsity
        optmAll = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(costAll)

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    sess = tf.Session(config=config)

    if restore is False:
        sess.run(init)
#        saver.restore(sess, '%s/60.cpkt' % modelDir)
    else:
        saver.restore(sess, '%s/%d.cpkt' % (modelDir, ncpkt))

    var = tf.trainable_variables()
    for i in var:
        print(i)

    if restore is False:
        return sess, saver, Xinput, Xoutput, Epoch_num, costMean, costSymmetric, costSparsity, optmAll, Yinput, prediction, lambdaStep, softThr, transField
    else:
        return sess, saver, Xinput, Xoutput, Yinput, Epoch_num, prediction, transField






# Add weight and bias
def add_conv2d_weight(wShape, nOrder):
    return tf.get_variable(shape=wShape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='weight_%d' % (nOrder))






def ista_block(layertensorY, layertensorZ, layerxk, layeryk, Phi, PhiT, Yinput, Epoch_num):

    # Parameters
    lambdaStep = tf.Variable(0.1, dtype=tf.float32)
    onec = tf.constant(1, dtype=tf.float32)
    softThr = tf.Variable(0.001, dtype=tf.float32)
    t = tf.Variable(1, dtype=tf.float32)
    channel=128
    convSize1 = channel
    convSize2 = channel
    convSize3 = channel
    filterSize1 = 3
    filterSize2 = 3
    filterSize3 = 3

    # rk <-- yk:1*120*160*3
    #rk = tf.subtract(layeryk[-1], tf.scalar_mul(lambdaStep, tf.multiply(Phi, tf.subtract(tf.multiply(Phi, layeryk[-1]), Yinput))))
    rho1=tf.div(onec, lambdaStep)
    Yrho=tf.scalar_mul(rho1,layertensorY[-1])
    Ak=layertensorZ[-1]-Yrho

    weight6 = add_conv2d_weight([filterSize1, filterSize1, nOfModel, 128], 6)
    Bk = tf.nn.conv2d(Ak, weight6, strides=[1, 1, 1, 1], padding='SAME')
    weight7 = add_conv2d_weight([filterSize1, filterSize1, 128, nOfModel], 7)
    Ck = tf.nn.conv2d(Bk, weight7, strides=[1, 1, 1, 1], padding='SAME')
    # Ck=Ak


    xk=tf.add(layerxk[0],tf.multiply(PhiT,Ck))
    rk=tf.add(xk,Yrho)

    # Drk <-- rk
    weight0 = add_conv2d_weight([filterSize1, filterSize1, nOfModel, convSize1], 0)
    Drk = tf.nn.conv2d(rk, weight0, strides=[1, 1, 1, 1], padding='SAME')

    # Transform Module
    weight1 = add_conv2d_weight([filterSize2, filterSize2, convSize1, convSize2], 1)
    weight2 = add_conv2d_weight([filterSize3, filterSize3, convSize2, convSize3], 2)
    Frk = tf.nn.conv2d(Drk, weight1, strides=[1, 1, 1, 1], padding='SAME')
    Frk = tf.nn.relu(Frk)
    Frk = tf.nn.conv2d(Frk, weight2, strides=[1, 1, 1, 1], padding='SAME')
    field = Frk

    # Soft-thresholding Module
    softFRk = tf.multiply(tf.sign(Frk), tf.nn.relu(tf.subtract(tf.abs(Frk), softThr)))


    # Inverse Transform Module
    weight3 = add_conv2d_weight([filterSize3, filterSize3, convSize3, convSize2], 3)
    weight4 = add_conv2d_weight([filterSize2, filterSize2, convSize2, convSize1], 4)
    FFrk = tf.nn.conv2d(softFRk, weight3, strides=[1, 1, 1, 1], padding='SAME')
    FFrk = tf.nn.relu(FFrk)
    FFrk = tf.nn.conv2d(FFrk, weight4, strides=[1, 1, 1, 1], padding='SAME')

    # get Grk from Rk
    weight5 = add_conv2d_weight([filterSize1, filterSize1, convSize1, nOfModel], 5)
    Grk = tf.nn.conv2d(FFrk, weight5, strides=[1, 1, 1, 1], padding='SAME')
    # xk = rk + Grk

    # xk = tf.add(rk, Grk)
    # yk = (1 + t)*xk - t*layerxk[-1]
    zk=Grk
    yk=tf.add(layertensorY[-1],tf.scalar_mul(lambdaStep, tf.subtract(xk,zk)))

    # Symmetric Restriction for Computing Loss Function
    sFrk = tf.nn.conv2d(Frk, weight3, strides=[1, 1, 1, 1], padding='SAME')
    sFrk = tf.nn.relu(sFrk)
    sFrk = tf.nn.conv2d(sFrk, weight4, strides=[1, 1, 1, 1], padding='SAME')

    # rk difference
    symmetric = sFrk - Drk  #转换到频域前、频域后再逆变换回来，的区别。不考虑软阈值
    return xk, yk, zk, symmetric, field, lambdaStep, softThr









def inference_ista(Y, Z, Xinput, Xoutput, Phi, PhiT, Yinput, Epoch_num, reuse):
    layerxk = []
    layeryk = []
    layerSymmetric = []
    transField = []
    layertensorY=[]
    layertensorZ = []
    layertensorY.append(Y)
    layertensorZ.append(Z)

    layerxk.append(Xinput)
    layeryk.append(Xinput)
    for i in range(nPhase):
        with tf.variable_scope('conv_%d' % (i), reuse=reuse):
            xk, yk, zk, convSymmetric, field, lambdaStep, softThr = ista_block(layertensorY, layertensorZ, layerxk, layeryk, Phi, PhiT, Yinput, Epoch_num)
            layerxk.append(xk)
            # layeryk.append(yk)
            layertensorY.append(yk)
            layertensorZ.append(zk)
            layerSymmetric.append(convSymmetric)
            transField.append(field)
    return layerxk, layerSymmetric, transField, lambdaStep, softThr








# Cost Computation
def compute_cost(prediction, predictionSymmetric, Xoutput, phi):
    phic=1-phi
    costMean = tf.reduce_mean(tf.square(tf.multiply(phic,prediction[-1]) - tf.multiply(phic,Xoutput)))
    costSymmetric = 0
    for k in range(nPhase):
        costSymmetric += tf.reduce_mean(tf.square(predictionSymmetric[k]))
    return costMean, costSymmetric






# Sparsity Computation
def compute_sparsity(tensor):
    costSparsity = 0
    for k in range(nPhase):
        costSparsity += tf.reduce_mean(tf.abs(tensor[k]))                 # l1 norm
    return costSparsity

