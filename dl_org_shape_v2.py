import cv2
import numpy as np
import os
import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as B
import tensorflow.keras.models as M
import tensorflow.keras.optimizers as O
import net_org_v2 as net
import tensorflow.keras.losses as Ls


route_all = 'D:/VO_ver.2/'
route_save = 'D:/dl_models/net_org_shape_v2_alternate_64.hdf5'
batch_size = 12
n_img =5
h=512
w=512
input = keras.Input((h,w,64))  #for 2d
#input = keras.Input((h,w,128,1)) #for 3d
output = net.organoid_net_2d_alternate(input)
#output = net.organoid_net_3d(input)
data_train = net.organoid_shape_data_large(route_all,split='train',batch_size=batch_size,height=h,width=w,n_img=n_img,dim=2)
data_val = net.organoid_shape_data_large(route_all,split='val',batch_size=batch_size,height=h,width=w,n_img=n_img,dim=2)
#output_5 = net.bosniak_net_3d_all_5out(input)
#data_train_5 = net.bosniak_data_3d_kc_integrate_5out(route_all)

def schedule(epoch,lr):
    if epoch%10==0 and epoch>4:
        return lr/2.0
    else:
        return lr

def jac_loss(y_true,y_pred):
    smooth = 0.00000001
    yt_r = y_true
    yp_r = y_pred

    intersection = B.sum(yt_r * yp_r)  # 이게 제일 생각해봐야 할 껀덕지 같은데
    union = B.sum(B.maximum(yt_r,yp_r))
    jaccard = (intersection) / (union + smooth)
#    dice = (2*intersection) /
#    print(intersection.shape, union.shape, jaccard.shape)
#    print(B.eval(intersection), B.eval(union), B.eval(jaccard))
    return 1-jaccard

def composite_loss(y_true,y_pred):
    smooth = 0.00000001
    yt_r = y_true
    yp_r = y_pred

    intersection = B.sum(yt_r * yp_r)  # 이게 제일 생각해봐야 할 껀덕지 같은데
    union = B.sum(B.maximum(yt_r,yp_r))
    jaccard = (intersection) / (union+ smooth)
    return (1-jaccard)
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
call_lr = keras.callbacks.LearningRateScheduler(schedule)
#acc_bi = keras.metrics.binary_accuracy()
model_org_shape = M.Model(inputs = input, outputs = output)
model_org_shape.compile(optimizer=O.Adam(learning_rate=0.001,decay=0.0001),loss = log_cosh_loss,metrics=['accuracy'])
model_org_shape.summary()
m_check = keras.callbacks.ModelCheckpoint(save_best_only=True,filepath='D:/dl_models/dl_org_shape_v2_alternate_64_progress.{epoch:02d}-{val_loss:.5f}.hdf5')
model_org_shape.fit_generator(generator=data_train,validation_data=data_val,epochs=100,shuffle=True,callbacks=[m_check,call_lr])
model_org_shape.save(route_save)