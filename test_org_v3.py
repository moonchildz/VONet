import cv2
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as B
import tensorflow.keras.models as M
import tensorflow.keras.optimizers as O
import tensorflow.keras.losses as Ls
import tensorflow_addons.layers as L_
import tensorflow_addons.optimizers as O_
import sys
print (sys.maxunicode)
def diceloss(y_true, y_pred):
    #y_pred = B.constant(y_pred)
    # flatten label and prediction tensors
    smooth = 0.00000001
    yt_r = y_true
    yp_r = y_pred

    intersection = B.mean(yt_r * yp_r) # 이게 제일 생각해봐야 할 껀덕지 같은데
    union = B.mean(yt_r) + B.mean(yp_r)
    dice = (2 * intersection + smooth) / (union + smooth)
    dice_ = (intersection + smooth) / (union - intersection + smooth)
    print(intersection,union,dice)
    return 1 - dice
def composite_loss(y_true,y_pred):
    smooth = 0.00000001
    yt_r = y_true
    yp_r = y_pred

    intersection = B.sum(yt_r * yp_r)  # 이게 제일 생각해봐야 할 껀덕지 같은데
    union = B.sum(B.maximum(yt_r,yp_r))
    jaccard = (intersection) / (B.sum(yt_r) + B.sum(yp_r) - intersection + smooth)
    absdiff = B.mean(B.abs(yt_r - yp_r))
    return (1-jaccard)*0.7 + absdiff*0.3

def jac_loss(y_true,y_pred):
    smooth = 0.00000001
    yt_r = y_true
    yp_r = y_pred

    intersection = B.sum(yt_r * yp_r)  # 이게 제일 생각해봐야 할 껀덕지 같은데
    union = B.sum(B.maximum(yt_r,yp_r))
    jaccard = (intersection) / (union + smooth)
    sum = B.mean(B.abs(yt_r - yp_r))
#    dice = (2*intersection) /
#    print(intersection.shape, union.shape, jaccard.shape)
#    print(B.eval(intersection), B.eval(union), B.eval(jaccard))
    return 1-jaccard + sum
def log_cosh_loss(y_true,y_pred):
    yt_r = y_true
    yp_r = y_pred
    smooth = 0.00000001
    intersection = B.sum(yt_r * yp_r)  # 이게 제일 생각해봐야 할 껀덕지 같은데
    tsum = B.sum(yt_r)
    psum = B.sum(yp_r)
    dice = 1- (2*intersection + 1 )/(tsum + psum + 1 )
    cosh = (B.exp(dice) + B.exp(-1*dice))/2.0
    logcosh = B.log(cosh)
    return logcosh
route_net_shape= 'D:/dl_models/dl_org_shape_128_progress.22-0.47054.hdf5'
def make_model(mode):
    if mode =='shape':
        org_net = M.load_model(route_net_shape, compile=True, custom_objects={'B': tf.keras.backend,'jac_loss':jac_loss})#
        org_net.summary()
    elif mode == 'livedead' or mode == 'ld':
        org_net = M.load_model(route_net_livedead, compile=True, custom_objects={'B': tf.keras.backend})#
        org_net.summary()
    return org_net
def derive_shape(org_net,img_array): #input " 1 x 256 x 256 x 64
    res = org_net.predict(img_array)
    return res
def derive_ldrate(org_net,img_array): #input " 1 x 256 x 256 x 5(variable at future)
    res = org_net.predict(img_array)
    return res
