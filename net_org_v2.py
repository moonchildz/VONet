import cv2
import numpy as np
import os
import tensorflow.keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as B
import random as rnd
import tensorflow.keras.initializers as I
from tensorflow.keras.utils import Sequence as Seq
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

layer_init = I.he_uniform()
f_init=32
sc01 = L.Conv2D(filters=f_init,kernel_size=(7,7),strides=(1,1),padding='same',name='sc01',kernel_initializer=layer_init)
sn01 = L.BatchNormalization(name='sb01')
sl01 = L.LeakyReLU(0.05,name='sl01')
sc02 = L.Conv2D(filters=f_init,kernel_size=(3,3),strides=(1,1),padding='same',name='sc02',kernel_initializer=layer_init)
sn02 = L.BatchNormalization(name='sb02')
sl02 = L.LeakyReLU(0.05,name='sl02')
sc03 = L.Conv2D(filters=f_init,kernel_size=(3,3),strides=(1,1),padding='same',name='sc03',kernel_initializer=layer_init)
sn03 = L.BatchNormalization(name='sb03')
sl03 = L.LeakyReLU(0.05,name='sl03')

ss01 = L.Conv2D(filters=f_init,kernel_size=(1,1),strides=(1,1),padding='same',name='ss01',kernel_initializer=layer_init)

sc04 = L.Conv2D(filters=f_init*2,kernel_size=(3,3),strides=(2,2),padding='same',name='sc04',kernel_initializer=layer_init)
sn04 = L.BatchNormalization(name='sb04')
sl04 = L.LeakyReLU(0.05,name='sl04')
sc05 = L.Conv2D(filters=f_init*2,kernel_size=(3,3),strides=(1,1),padding='same',name='sc05',kernel_initializer=layer_init)
sn05 = L.BatchNormalization(name='sb05')
sl05 = L.LeakyReLU(0.05,name='sl05')
sc06 = L.Conv2D(filters=f_init*2,kernel_size=(3,3),strides=(1,1),padding='same',name='sc06',kernel_initializer=layer_init)
sn06 = L.BatchNormalization(name='sb06')
sl06 = L.LeakyReLU(0.05,name='sl06')

ss02 = L.Conv2D(filters=f_init*2,kernel_size=(1,1),strides=(1,1),padding='same',name='ss02',kernel_initializer=layer_init)

sc07 = L.Conv2D(filters=f_init*4,kernel_size=(3,3),strides=(2,2),padding='same',name='sc07')
sn07 = L.BatchNormalization(name='sb07')
sl07 = L.LeakyReLU(0.05,name='sl07')
sc08 = L.Conv2D(filters=f_init*4,kernel_size=(3,3),strides=(1,1),padding='same',name='sc08')
sn08 = L.BatchNormalization(name='sb08')
sl08 = L.LeakyReLU(0.05,name='sl08')
sc09 = L.Conv2D(filters=f_init*4,kernel_size=(3,3),strides=(1,1),padding='same',name='sc09')
sn09 = L.BatchNormalization(name='sb09')
sl09 = L.LeakyReLU(0.05,name='sl09')

ss03 = L.Conv2D(filters=f_init*4,kernel_size=(1,1),strides=(1,1),padding='same',name='ss03')

sc10 = L.Conv2D(filters=f_init*8,kernel_size=(3,3),strides=(2,2),padding='same',name='sc10')
sn10 = L.BatchNormalization(name='sb10')
sl10 = L.LeakyReLU(0.05,name='sl10')
sc11 = L.Conv2D(filters=f_init*8,kernel_size=(3,3),strides=(1,1),padding='same',name='sc11')
sn11 = L.BatchNormalization(name='sb11')
sl11 = L.LeakyReLU(0.05,name='sl11')
sc12 = L.Conv2D(filters=f_init*8,kernel_size=(3,3),strides=(1,1),padding='same',name='sc12')
sn12 = L.BatchNormalization(name='sb12')
sl12 = L.LeakyReLU(0.05,name='sl12')
##################################################################################### downsampling cnn
su01 = L.UpSampling2D(size=(2,2),name = 'su01')
sco1 = L.Concatenate(axis=-1,name = 'sco1')
sc13 = L.Conv2D(filters=f_init*4,kernel_size=(3,3),strides=(1,1),padding='same',name='sc13')
sn13 = L.BatchNormalization(name='sb13')
sl13 = L.LeakyReLU(0.05,name='sl13')
sc14 = L.Conv2D(filters=f_init*4,kernel_size=(3,3),strides=(1,1),padding='same',name='sc14')
sn14 = L.BatchNormalization(name='sb14')
sl14 = L.LeakyReLU(0.05,name='sl14')
sc15 = L.Conv2D(filters=f_init*4,kernel_size=(3,3),strides=(1,1),padding='same',name='sc15')
sn15 = L.BatchNormalization(name='sb15')
sl15 = L.LeakyReLU(0.05,name='sl15')        #0.25
ans_025 = L.Conv2D(filters=1,kernel_size=(3,3),strides=1,padding='same',name='ans025')

su02 = L.UpSampling2D(size=(2,2),name = 'su02')
sco2 = L.Concatenate(axis=-1,name = 'sco2')
sc16 = L.Conv2D(filters=f_init*2,kernel_size=(3,3),strides=(1,1),padding='same',name='sc16')
sn16 = L.BatchNormalization(name='sb16')
sl16 = L.LeakyReLU(0.05,name='sl16')
sc17 = L.Conv2D(filters=f_init*2,kernel_size=(3,3),strides=(1,1),padding='same',name='sc17')
sn17 = L.BatchNormalization(name='sb17')
sl17 = L.LeakyReLU(0.05,name='sl17')
sc18 = L.Conv2D(filters=f_init*2,kernel_size=(3,3),strides=(1,1),padding='same',name='sc18')
sn18 = L.BatchNormalization(name='sb18')
sl18 = L.LeakyReLU(0.05,name='sl18')
ans_05 = L.Conv2D(filters=1,kernel_size=(3,3),strides=1,padding='same',name='ans05')

su03 = L.UpSampling2D(size=(2,2),name = 'su03')
sco3 = L.Concatenate(axis=-1,name = 'sco3')
sc19 = L.Conv2D(filters=f_init,kernel_size=(3,3),strides=(1,1),padding='same',name='sc19')
sn19 = L.BatchNormalization(name='sb19')
sl19 = L.LeakyReLU(0.05,name='sl19')
sc20 = L.Conv2D(filters=f_init,kernel_size=(3,3),strides=(1,1),padding='same',name='sc20')
sn20 = L.BatchNormalization(name='sb20')
sl20 = L.LeakyReLU(0.05,name='sl20')
sc21 = L.Conv2D(filters=f_init,kernel_size=(3,3),strides=(1,1),padding='same',name='sc21')
sn21 = L.BatchNormalization(name='sb21')
sl21 = L.LeakyReLU(0.05,name='sl21')
ans_1 = L.Conv2D(filters=1,kernel_size=(3,3),strides=1,padding='same',name='ans1')



class organoid_shape_data(Seq):
    def __init__(self,route,batch_size = 4,split ='train', height = 512, width = 512,n_img = 7,dim=2,shuffle=False):
        self.route_main = route
        self.batch_size = batch_size
        self.split = split
        self.n_img = n_img
        self.dim = dim
        self.shuffle = shuffle
        self.organoid_route= 'D:/VO_ver.2/'
        self.h = height
        self.w = width
        self.org_list  = os.listdir(self.organoid_route)
        if self.shuffle ==True:
            self.org_list = rnd.shuffle(self.org_list)

        cut = int(0.9*len(self.org_list))
        if self.split == 'train':
            self.pl = self.org_list[:cut]
        elif self.split == 'val':
            self.pl = self.org_list[cut:]
       # for io, o in enumerate(self.org_list):



        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.pl) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        org_list_temp = [self.pl[k] for k in indexes]
        # Generate data
        I,O = self.__data_generation(org_list_temp)
        return I,O

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.pl))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    #route_main = D:/ps dataset/PS_Sculpture_Dataset/
    def __data_generation(self,org_list_temp):
        img_selected = []
        vox_selected = []
        for io,o in enumerate(org_list_temp):
            voxel_route = self.organoid_route+o+'/voxel_'+o+'.xyz'
            vox = open(voxel_route,'r').readlines()[1:-2]
            vox_np = np.zeros((64,64,64))
            vox_np_neg = np.ones((64,64,64))
            img_stack_np = np.zeros((self.h,self.w,256))
            for v in vox:
                vline = v.split(' ')[1:]
                vox_np[int(vline[0]), int(vline[1]), int(vline[2])] = 1
            vox_np_neg[vox_np==1] =0
            #이러면 Output GT는 완성이 됐고
            img_route = self.organoid_route+o+'/img/'
            imgf_route = self.organoid_route+o+'/focusmap_05/'
            imglist = os.listdir(img_route)
            num_img = rnd.randint(1,self.n_img)
            step = int(88 / (num_img + 1) )
            img_seq = []

            for n in range(1, num_img + 1):
                img_seq.append(n * step + rnd.randint(-3, 3)+20)
            for r in img_seq:
             #   print(img_seq, r)
                img_rth = imglist[r]
                imgf_rth = img_rth.split('.')[0]+'.png'
                img = cv2.imread(img_route+img_rth,-1)/255.0
                imgf = cv2.imread(imgf_route+imgf_rth,-1)/255.0
                img = cv2.resize(img,(512,512))
                #imgf[imgf<0.8] = 0
                img_stack_np[:,:,r]=img*imgf
            img_selected.append(img_stack_np)

            vox_selected.append(vox_np)
        img_np = np.array(img_selected)
        vox_np = np.array(vox_selected)
      #  print('  Shape of batch:', img_np.shape, vox_np.shape,self.n_img)
        #img_np = img_np.transpose(2, 0, 1)
        if self.dim == 3:
            img_np = np.expand_dims(img_np,-1)  #if 3-dim : (w,h,128,1) elif 2-dim : (w,h,128)

        return img_np, vox_np

class organoid_shape_data_large(Seq):
    def __init__(self,route,batch_size = 4,split ='train', height = 512, width = 512,n_img = 5,dim=2,shuffle=False):
        self.route_main = route
        self.batch_size = batch_size
        self.split = split
        self.n_img = n_img
        self.dim = dim
        self.shuffle = shuffle
        self.organoid_route= 'D:/VO_ver.2/'
        self.h = height
        self.w = width
        self.org_list  = os.listdir(self.organoid_route)
        if self.shuffle ==True:
            self.org_list = rnd.shuffle(self.org_list)

        cut = int(0.9*len(self.org_list))
        if self.split == 'train':
            self.pl = self.org_list[:cut]
        elif self.split == 'val':
            self.pl = self.org_list[cut:]
       # for io, o in enumerate(self.org_list):



        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.pl) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        org_list_temp = [self.pl[k] for k in indexes]
        # Generate data
        I,O = self.__data_generation(org_list_temp)
        return I,O

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.pl))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    #route_main = D:/ps dataset/PS_Sculpture_Dataset/
    def __data_generation(self,org_list_temp):
        img_selected = []
        vox_selected = []
        for io,o in enumerate(org_list_temp):
            voxel_route = self.organoid_route+o+'/voxel_large_'+o+'.xyz'
            #vox_center_route = self.organoid_route+o+'/voxel_'+o+'.xyz'
            vox = open(voxel_route,'r').readlines()[1:-2]
            #vox_center = open(voxel_center_route,'r').readlines()[1:-2]
            vox_np = np.zeros((512,512,64))
            #vox_center_np = np.zeros((128,128,128))
            img_stack_np = np.zeros((self.h,self.w,64))


           # for v in vox:
           #     vline = v.split(' ')[1:]
           #     vox_np[int(vline[0]), int(vline[1]), int(vline[2])//2] = 1  # 128 x 128 x 32 를 맞추기 위해서
            #for v in vox_center:
            #    vline = v.split(' ')[1:]
            #    vox_center_np[int(vline[0])*2, int(vline[1])*2, int(vline[2])*2] = 1

#            vox_np_neg[vox_np==1] =0
            #이러면 Output GT는 완성이 됐고
            img_route = self.organoid_route+o+'/img/'
            imgf_route = self.organoid_route+o+'/img_focus/'
            imglist = os.listdir(img_route)
            vox_list = []
            for i in range(64):
                img_ith = str(i*2+1)+'.jpg'
                imgf = cv2.resize(cv2.imread(imgf_route + img_ith,0)/255,(128,128))
                imgf[imgf>0.1]=1
                imgf[imgf<=0.1]=0
                vox_list.append(imgf)
            vox_list = np.array(vox_list)
            vox_list = np.transpose(vox_list,[1,2,0])

            num_img = rnd.randint(5,31)
            step = int(108 / (num_img + 1) )
            img_seq = []
            # 오늘인 원래 128 쓰던걸 32 만 쓸거에요 ~
            for n in range(1, num_img + 1):
                #imgnum = 15 + 113 -> 4 ~ 28
                img_seq.append((n * step + 10) + rnd.randint(-1,1))   #이거 rnd 안 넣어봐야겠다 어케되나
            for r in img_seq:
             #   print(img_seq, r)
                img_rth = str(r)+'.jpg'
                imgf_rth = str(r)+'.png'
                img = cv2.imread(img_route+img_rth,-1)/255.0
                img = cv2.resize(img,(512,512))
                #imgf[imgf<0.8] = 0
                img_stack_np[:,:,r//2]=img
            img_selected.append(img_stack_np)

            vox_selected.append(vox_list)
        img_np = np.array(img_selected)
        vox_np = np.array(vox_selected)
        #print('  Shape of batch:', img_np.shape, vox_np.shape,num_img)
        #img_np = img_np.transpose(2, 0, 1)

        return img_np, vox_np

class organoid_shape_data_verylarge(Seq):
    def __init__(self,route,batch_size = 4,split ='train', height = 512, width = 512,n_img = 7,dim=2,shuffle=False):
        self.route_main = route
        self.batch_size = batch_size
        self.split = split
        self.n_img = n_img
        self.dim = dim
        self.shuffle = shuffle
        self.organoid_route= 'D:/VO_ver.2/'
        self.h = height
        self.w = width
        self.org_list  = os.listdir(self.organoid_route)
        if self.shuffle ==True:
            self.org_list = rnd.shuffle(self.org_list)

        cut = int(0.9*len(self.org_list))
        if self.split == 'train':
            self.pl = self.org_list[:cut]
        elif self.split == 'val':
            self.pl = self.org_list[cut:]
       # for io, o in enumerate(self.org_list):



        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.pl) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        org_list_temp = [self.pl[k] for k in indexes]
        # Generate data
        I,O = self.__data_generation(org_list_temp)
        return I,O

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.pl))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    #route_main = D:/ps dataset/PS_Sculpture_Dataset/
    def __data_generation(self,org_list_temp):
        img_selected = []
        vox_selected = []
        for io,o in enumerate(org_list_temp):
            voxel_route = self.organoid_route+o+'/voxel_'+o+'.xyz'
            vox = open(voxel_route,'r').readlines()[1:-2]
            vox_np = np.zeros((64,64,64))
            vox_np_neg = np.ones((64,64,64))
            img_stack_np = np.zeros((self.h,self.w,256))
            for v in vox:
                vline = v.split(' ')[1:]
                vox_np[int(vline[0]), int(vline[1]), int(vline[2])] = 1
            vox_np_neg[vox_np==1] =0
            #이러면 Output GT는 완성이 됐고
            img_route = self.organoid_route+o+'/focusmap_05/'
            imglist = os.listdir(img_route)
            num_img = rnd.randint(1,self.n_img)
            step = int(88 / (num_img + 1) )
            img_seq = []

            for n in range(1, num_img + 1):
                img_seq.append(n * step + rnd.randint(-3, 3)+20)
            for r in img_seq:
             #   print(img_seq, r)
                img_rth = imglist[r]
                imgf_rth = img_rth.split('.')[0]+'.png'
                img = cv2.imread(img_route+img_rth,-1)/255.0
                img = cv2.resize(img,(64,64))
                #imgf[imgf<0.8] = 0
                img_stack_np[:,:,r]=img
            img_selected.append(img_stack_np)

            vox_selected.append(vox_np)
        img_np = np.array(img_selected)
        vox_np = np.array(vox_selected)
      #  print('  Shape of batch:', img_np.shape, vox_np.shape,self.n_img)
        #img_np = img_np.transpose(2, 0, 1)
        if self.dim == 3:
            img_np = np.expand_dims(img_np,-1)  #if 3-dim : (w,h,128,1) elif 2-dim : (w,h,128)

        return img_np, vox_np
class organoid_shape_data_3d(Seq):
    def __init__(self,route,batch_size = 4,split ='train', height = 512, width = 512,n_img = 7,dim=3,shuffle=False):
        self.route_main = route
        self.batch_size = batch_size
        self.split = split
        self.n_img = n_img
        self.dim = dim
        self.shuffle = shuffle
        self.organoid_route= 'D:/VO_ver.2/'
        self.h = height
        self.w = width
        self.org_list  = os.listdir(self.organoid_route)
        if self.shuffle ==True:
            self.org_list = rnd.shuffle(self.org_list)

        cut = int(0.9*len(self.org_list))
        if self.split == 'train':
            self.pl = self.org_list[:cut]
        elif self.split == 'val':
            self.pl = self.org_list[cut:]
       # for io, o in enumerate(self.org_list):



        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.pl) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        org_list_temp = [self.pl[k] for k in indexes]
        # Generate data
        I,O = self.__data_generation(org_list_temp)
        return I,O

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.pl))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    #route_main = D:/ps dataset/PS_Sculpture_Dataset/
    def __data_generation(self,org_list_temp):
        img_selected = []
        vox_selected = []
        for io,o in enumerate(org_list_temp):
            voxel_route = self.organoid_route+o+'/voxel_'+o+'.xyz'
            voxel_large_route = self.organoid_route+o+'/voxel_large_'+o+'.xyz'
            vox = open(voxel_route,'r').readlines()[1:-2]
            vox_large = open(voxel_large_route,'r').readlines()[1:-2]
            vox_np = np.zeros((128,128,128,1))
            vox_np_center = np.zeros((128,128,128,1))
            vox_np_neg = np.ones((128,128,128,1))
            img_stack_np = np.zeros((self.h,self.w,128,1))
            for v in vox:
                vline = v.split(' ')[1:]
                vox_np_center[int(vline[0])*2, int(vline[1])*2, int(vline[2])*2] = 1
            for v in vox_large:
                vline = v.split(' ')[1:]
                vox_np[int(vline[0]), int(vline[1]), int(vline[2])] = 1
            vox_np[vox_np_center==1] =0
            vox_np_neg[vox_np==1] =0
            vox_stack_np = np.concatenate([vox_np,vox_np_center,vox_np_neg],-1)
            #이러면 Output GT는 완성이 됐고
            img_route = self.organoid_route+o+'/img/'
            imglist = os.listdir(img_route)
            num_img = rnd.randint(1,self.n_img)
            step = int(128 / (num_img + 1) )
            img_seq = []

            for n in range(1, num_img + 1):
                img_seq.append(n * step + rnd.randint(-3, 3))
            for r in img_seq:
             #   print(img_seq, r)

                img_rth = str(r)+'.jpg'
                img = np.expand_dims(cv2.imread(img_route+img_rth,0)/255.0,-1)
                img = cv2.resize(img,(256,256))
                img_stack_np[:,:,r,0]=img

            img_selected.append(img_stack_np)

            vox_selected.append(vox_np)
        img_np = np.array(img_selected)
        vox_np = np.array(vox_selected)

        return img_np, vox_np

def focusnet(input):
    def relu_end(x):
        r = B.relu(x,max_value=1)
        return r

    x1 = sc01(input)
    x1 = sn01(x1)
    x1 = sl01(x1)
    x1 = sc02(x1)
    x1 = sn02(x1)
    x1 = sl02(x1)
    x1 = sc03(x1)
    x1 = sn03(x1)
    x1 = sl03(x1)
    xs1 = ss01(x1)

    x2 = sc04(x1)
    x2 = sn04(x2)
    x2 = sl04(x2)
    x2 = sc05(x2)
    x2 = sn05(x2)
    x2 = sl05(x2)
    x2 = sc06(x2)
    x2 = sn06(x2)
    x2 = sl06(x2)
    xs2 = ss02(x2)

    x3 = sc07(x2)
    x3 = sn07(x3)
    x3 = sl07(x3)
    x3 = sc08(x3)
    x3 = sn08(x3)
    x3 = sl08(x3)
    x3 = sc09(x3)
    x3 = sn09(x3)
    x3 = sl09(x3)
    xs3 = ss03(x3)

    x4 = sc10(x3)
    x4 = sn10(x4)
    x4 = sl10(x4)
    x4 = sc11(x4)
    x4 = sn11(x4)
    x4 = sl11(x4)
    x4 = sc12(x4)
    x4 = sn12(x4)
    x4 = sl12(x4)

    xu3 = su01(x4)
    xc3 = sco1([xs3,xu3])
    x3_ = sc13(xc3)
    x3_ = sn13(x3_)
    x3_ = sl13(x3_)
    x3_ = sc14(x3_)
    x3_ = sn14(x3_)
    x3_ = sl14(x3_)
    x3_ = sc15(x3_)
    x3_ = sn15(x3_)
    x3_ = sl15(x3_)
    xa_025 = ans_025(x3_)

    xu2 = su02(x3_)
    xc2 = sco2([xs2,xu2])
    x2_ = sc13(xc2)
    x2_ = sn13(x2_)
    x2_ = sl13(x2_)
    x2_ = sc14(x2_)
    x2_ = sn14(x2_)
    x2_ = sl14(x2_)
    x2_ = sc15(x2_)
    x2_ = sn15(x2_)
    x2_ = sl15(x2_)
    xa_05 = ans_05(x2_)

    xu1 = su03(x2_)
    xc1 = sco3([xs1,xu1])
    x1_ = sc16(xc1)
    x1_ = sn16(x1_)
    x1_ = sl16(x1_)
    x1_ = sc17(x1_)
    x1_ = sn17(x1_)
    x1_ = sl17(x1_)
    x1_ = sc18(x1_)
    x1_ = sn18(x1_)
    x1_ = sl18(x1_)
    xa_1 = ans_1(x1_)

    xa_1 = L.Lambda(relu_end)(xa_1)
    xa_05 = L.Lambda(relu_end)(xa_05)
    xa_025 = L.Lambda(relu_end)(xa_025)
    return xa_1,xa_05,xa_025
def organoid_net_2d(input):    #in : 512 x 512 x 128 or 256 x 256 x 128

    x0 = L.Conv2D(filters=32, kernel_size=(7, 7), strides=1, padding='same')(input)  # 512 x 512 x 32
    x0 = L.BatchNormalization()(x0)
    x0 = L.LeakyReLU(0.1)(x0)
    x0 = L.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same')(x0)  # 512 x 512 x 32
    x0 = L.BatchNormalization()(x0)
    x0 = L.LeakyReLU(0.1)(x0)
    x0 = L.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same')(x0)  # 512 x 512 x 32
    x0 = L.BatchNormalization()(x0)
    x0 = L.LeakyReLU(0.1)(x0)
   # x0 = L.Dropout(0.1)(x0)

    x1 = L.Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding='same')(x0) #256 x 256 x 64
    x1 = L.BatchNormalization()(x1)
    x1 = L.LeakyReLU(0.1)(x1)
    x1 = L.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x1)
    x1 = L.BatchNormalization()(x1)
    x1 = L.LeakyReLU(0.1)(x1)
    x1 = L.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x1)
    x1 = L.BatchNormalization()(x1)
    x1 = L.LeakyReLU(0.1)(x1)
   # x1 = L.Dropout(0.1)(x1)

    x2 = L.Conv2D(filters=128, kernel_size=(5, 5), strides=2, padding='same')(x1) #128 x 128 x 128
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(0.1)(x2)
    x2 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x2)
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(0.1)(x2)
    x2 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x2)
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(0.1)(x2)
  #  x2 = L.Dropout(0.1)(x2)

    x3 = L.Conv2D(filters=256, kernel_size=(5, 5), strides=2, padding='same')(x2) #64 x 64 x 256
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
    x3 = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x3)
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
    x3 = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x3)
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
   # x3 = L.Dropout(0.1)(x3)

    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding='same')(x3) #32 x 32 x 512
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)

# x4 = L.Dropout(0.1)(x4)

    xu = L.UpSampling2D(size=(2,2))(x4)                                           #64 x 64 x 512
    x3_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(xu)#64 x 64 x 256
    x3_ = L.BatchNormalization()(x3_)
    x3_ = L.LeakyReLU(0.1)(x3_)

    xc = L.Concatenate(axis=-1)([x3,x3_])
    x4_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(xc)#64 x 64 x 256
    x4_ = L.BatchNormalization()(x4_)
    x4_ = L.LeakyReLU(0.1)(x4_)
    x4_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x4_)#64 x 64 x 256
    x4_ = L.BatchNormalization()(x4_)
    x4_ = L.LeakyReLU(0.1)(x4_)
    x4_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x4_)#64 x 64 x 64
    x4_ = L.BatchNormalization()(x4_)
    x4_ = L.LeakyReLU(0.1)(x4_)
    x4_ = L.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x4_)
    #x_relu = L.ReLU()(x4_)
    def relu_end(x):
        r = B.relu(x,max_value=1)
        return r
    x_relu = L.Lambda(relu_end)(x4_)
    return x_relu

def organoid_net_2d_verysmall(input):    #in : 512 x 512 x 128 or 256 x 256 x 128

    x0 = L.Conv2D(filters=32*2, kernel_size=(5, 5), strides=1, padding='same')(input)  # 64 x 64 x 64
    x0 = L.BatchNormalization()(x0)
    x0 = L.LeakyReLU(0.1)(x0)
    x0 = L.Conv2D(filters=32*2, kernel_size=(3, 3), strides=1, padding='same')(x0)  # 64 x 64 x 64
    x0 = L.BatchNormalization()(x0)
    x0 = L.LeakyReLU(0.1)(x0)
    x0 = L.Conv2D(filters=32*2, kernel_size=(3, 3), strides=1, padding='same')(x0)  # 64 x 64 x 64
    x0 = L.BatchNormalization()(x0)
    x0 = L.LeakyReLU(0.1)(x0)
    x0 = L.Dropout(0.1)(x0)

    x1 = L.Conv2D(filters=64*2, kernel_size=(5, 5), strides=2, padding='same')(x0) #32 x 32 x 128
    x1 = L.BatchNormalization()(x1)
    x1 = L.LeakyReLU(0.1)(x1)
    x1 = L.Conv2D(filters=64*2, kernel_size=(3, 3), strides=1, padding='same')(x1)
    x1 = L.BatchNormalization()(x1)
    x1 = L.LeakyReLU(0.1)(x1)
    x1 = L.Conv2D(filters=64*2, kernel_size=(3, 3), strides=1, padding='same')(x1)
    x1 = L.BatchNormalization()(x1)
    x1 = L.LeakyReLU(0.1)(x1)
    x1 = L.Dropout(0.1)(x1)

    x2 = L.Conv2D(filters=128*2, kernel_size=(5, 5), strides=2, padding='same')(x1) #16 x 16 x 256
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(0.1)(x2)
    x2 = L.Conv2D(filters=128*2, kernel_size=(3, 3), strides=1, padding='same')(x2)
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(0.1)(x2)
    x2 = L.Conv2D(filters=128*2, kernel_size=(3, 3), strides=1, padding='same')(x2)
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(0.1)(x2)
    x2 = L.Dropout(0.1)(x2)

    x3 = L.Conv2D(filters=256*2, kernel_size=(5, 5), strides=2, padding='same')(x2) #8 x 8 x 512
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
    x3 = L.Conv2D(filters=256*2, kernel_size=(3, 3), strides=1, padding='same')(x3)
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
    x3 = L.Conv2D(filters=256*2, kernel_size=(3, 3), strides=1, padding='same')(x3)
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
    x3 = L.Dropout(0.1)(x3)

    xu = L.UpSampling2D(size=(2,2))(x3)                                           #16 x 16 x 256
    x2_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(xu)
    x2_ = L.BatchNormalization()(x2_)
    x2_ = L.LeakyReLU(0.1)(x2_)

    xc = L.Concatenate(axis=-1)([x2,x2_])
    x2_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(xc)
    x2_ = L.BatchNormalization()(x2_)
    x2_ = L.LeakyReLU(0.1)(x2_)
    x2_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x2_)
    x2_ = L.BatchNormalization()(x2_)
    x2_ = L.LeakyReLU(0.1)(x2_)

    xu2 = L.UpSampling2D(size=(2,2))(x2_)                                           #32 x 32 x 128
    x1_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(xu2)
    x1_ = L.BatchNormalization()(x1_)
    x1_ = L.LeakyReLU(0.1)(x1_)

    xc2 = L.Concatenate(axis=-1)([x1,x1_])
    x1_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(xc2)
    x1_ = L.BatchNormalization()(x1_)
    x1_ = L.LeakyReLU(0.1)(x1_)
    x1_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x1_)
    x1_ = L.BatchNormalization()(x1_)
    x1_ = L.LeakyReLU(0.1)(x1_)

    xu3 = L.UpSampling2D(size=(2,2))(x1_)                                           #64 x 64 x 64
    x0_ = L.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(xu3)
    x0_ = L.BatchNormalization()(x0_)
    x0_ = L.LeakyReLU(0.1)(x0_)

    xc3 = L.Concatenate(axis=-1)([x0,x0_])
    x0_ = L.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(xc3)
    x0_ = L.BatchNormalization()(x0_)
    x0_ = L.LeakyReLU(0.1)(x0_)
    x0_ = L.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x0_)
    x0_ = L.BatchNormalization()(x0_)
    x0_ = L.LeakyReLU(0.1)(x0_)
    x0_ = L.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x0_)

    def relu_end(x):
        r = B.relu(x,max_value=1)
        return r
    x_relu = L.Lambda(relu_end)(x0_)
    return x_relu

def organoid_net_2d_large(input):    #in : 512 x 512 x 128 or 256 x 256 x 128

    x0 = L.Conv2D(filters=32, kernel_size=(7, 7), strides=1, padding='same')(input)  # 512 x 512 x 32
    x0 = L.BatchNormalization()(x0)
    x0 = L.LeakyReLU(0.1)(x0)
    x0 = L.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same')(x0)  # 512 x 512 x 32
    x0 = L.BatchNormalization()(x0)
    x0 = L.LeakyReLU(0.1)(x0)
    x0 = L.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same')(x0)  # 512 x 512 x 32
    x0 = L.BatchNormalization()(x0)
    x0 = L.LeakyReLU(0.1)(x0)
    x0 = L.Dropout(0.1)(x0)

    x1 = L.Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding='same')(x0) #256 x 256 x 64
    x1 = L.BatchNormalization()(x1)
    x1 = L.LeakyReLU(0.1)(x1)
    x1 = L.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x1)
    x1 = L.BatchNormalization()(x1)
    x1 = L.LeakyReLU(0.1)(x1)
    x1 = L.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x1)
    x1 = L.BatchNormalization()(x1)
    x1 = L.LeakyReLU(0.1)(x1)
    x1 = L.Dropout(0.1)(x1)

    x2 = L.Conv2D(filters=128, kernel_size=(5, 5), strides=2, padding='same')(x1) #128 x 128 x 128
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(0.1)(x2)
    x2 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x2)
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(0.1)(x2)
    x2 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x2)
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(0.1)(x2)
    x2 = L.Dropout(0.1)(x2)

    x3 = L.Conv2D(filters=256, kernel_size=(5, 5), strides=2, padding='same')(x2) #64 x 64 x 256
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
    x3 = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x3)
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
    x3 = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x3)
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
    x3 = L.Dropout(0.1)(x3)

    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding='same')(x3) #32 x 32 x 512
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Dropout(0.1)(x4)

    xu = L.UpSampling2D(size=(2,2))(x4)                                           #64 x 64 x 512
    xc = L.Concatenate(axis=-1)([x3,xu])
    x3_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(xc)#64 x 64 x 256
    x3_ = L.BatchNormalization()(x3_)
    x3_ = L.LeakyReLU(0.1)(x3_)
    x3_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x3_)#64 x 64 x 256
    x3_ = L.BatchNormalization()(x3_)
    x3_ = L.LeakyReLU(0.1)(x3_)
    x3_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x3_)#64 x 64 x 256
    x3_ = L.BatchNormalization()(x3_)
    x3_ = L.LeakyReLU(0.1)(x3_)

    xu2 = L.UpSampling2D(size=(2, 2))(x3_)  # 64 x 64 x 512
    xc2 = L.Concatenate(axis=-1)([x2,xu2])
    x2_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(xc2)  # 128 x 128 x 128
    x2_ = L.BatchNormalization()(x2_)
    x2_ = L.LeakyReLU(0.1)(x2_)
    x2_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x2_)  # 128 x 128 x 128
    x2_ = L.BatchNormalization()(x2_)
    x2_ = L.LeakyReLU(0.1)(x2_)
    x2_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x2_)  # 128 x 128 x 128
    x2_ = L.BatchNormalization()(x2_)
    x2_ = L.LeakyReLU(0.1)(x2_)
    x2_ = L.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same')(x2_)  # 128 x 128 x 128

    #x_relu = L.ReLU()(x2_)
    def relu_end(x):
        r = B.relu(x,max_value=1)
        return r
    x_relu = L.Lambda(relu_end)(x2_)
    return x_relu

def organoid_net_2d_verylarge(input):    #in : 512 x 512 x 128 or 256 x 256 x 128

    x0 = L.Conv2D(filters=32, kernel_size=(7, 7), strides=1, padding='same',kernel_initializer=layer_init)(input)  # 512 x 512 x 32
    x0 = L.BatchNormalization()(x0)
    x0 = L.LeakyReLU(0.1)(x0)
    x0 = L.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x0)  # 512 x 512 x 32
    x0 = L.BatchNormalization()(x0)
    x0 = L.LeakyReLU(0.1)(x0)
    x0 = L.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x0)  # 512 x 512 x 32
    x0 = L.BatchNormalization()(x0)
    x0 = L.LeakyReLU(0.1)(x0)
    x0 = L.Dropout(0.05)(x0)

    x1 = L.Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding='same',kernel_initializer=layer_init)(x0) #256 x 256 x 64
    x1 = L.BatchNormalization()(x1)
    x1 = L.LeakyReLU(0.1)(x1)
    x1 = L.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x1)
    x1 = L.BatchNormalization()(x1)
    x1 = L.LeakyReLU(0.1)(x1)
    x1 = L.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x1)
    x1 = L.BatchNormalization()(x1)
    x1 = L.LeakyReLU(0.1)(x1)
    x1 = L.Dropout(0.05)(x1)

    x2 = L.Conv2D(filters=128, kernel_size=(5, 5), strides=2, padding='same',kernel_initializer=layer_init)(x1) #128 x 128 x 128
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(0.1)(x2)
    x2 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x2)
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(0.1)(x2)
    x2 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x2)
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(0.1)(x2)
    x2 = L.Dropout(0.05)(x2)

    x3 = L.Conv2D(filters=256, kernel_size=(5, 5), strides=2, padding='same',kernel_initializer=layer_init)(x2) #64 x 64 x 256
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
    x3 = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x3)
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
    x3 = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x3)
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
    x3 = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x3)
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
    x3 = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x3)
    x3 = L.BatchNormalization()(x3)
    x3 = L.LeakyReLU(0.1)(x3)
    x3 = L.Dropout(0.05)(x3)

    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding='same',kernel_initializer=layer_init)(x3) #32 x 32 x 512
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.LeakyReLU(0.1)(x4)
    x4 = L.Dropout(0.05)(x4)

    xu3 = L.UpSampling2D(size=(2,2))(x4)                                           #64 x 64 x 512
    xc3 = L.Concatenate(axis=-1)([x3,xu3])
    x3_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(xc3)#64 x 64 x 256
    x3_ = L.BatchNormalization()(x3_)
    x3_ = L.LeakyReLU(0.1)(x3_)
    x3_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x3_)#64 x 64 x 256
    x3_ = L.BatchNormalization()(x3_)
    x3_ = L.LeakyReLU(0.1)(x3_)
    x3_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x3_)#64 x 64 x 256
    x3_ = L.BatchNormalization()(x3_)
    x3_ = L.LeakyReLU(0.1)(x3_)
    x3_ = L.Dropout(0.05)(x3_)

    xu2 = L.UpSampling2D(size=(2, 2))(x3_)  # 64 x 64 x 512
    xc2 = L.Concatenate(axis=-1)([x2,xu2])
    x2_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(xc2)  # 128 x 128 x 128
    x2_ = L.BatchNormalization()(x2_)
    x2_ = L.LeakyReLU(0.1)(x2_)
    x2_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x2_)  # 128 x 128 x 128
    x2_ = L.BatchNormalization()(x2_)
    x2_ = L.LeakyReLU(0.1)(x2_)
    x2_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x2_)  # 128 x 128 x 128
    x2_ = L.BatchNormalization()(x2_)
    x2_ = L.LeakyReLU(0.1)(x2_)
    x2_ = L.Dropout(0.05)(x2_)

    xu1 = L.UpSampling2D(size=(2, 2))(x2_)  # 64 x 64 x 512
    xc1 = L.Concatenate(axis=-1)([x1, xu1])
    x1_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(xc1)  # 256 x 256 x 128
    x1_ = L.BatchNormalization()(x1_)
    x1_ = L.LeakyReLU(0.1)(x1_)
    x1_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x1_)  # 128 x 128 x 128
    x1_ = L.BatchNormalization()(x1_)
    x1_ = L.LeakyReLU(0.1)(x1_)
    x1_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same',kernel_initializer=layer_init)(x1_)  # 128 x 128 x 128
    x1_ = L.BatchNormalization()(x1_)
    x1_ = L.LeakyReLU(0.1)(x1_)
    x1_ = L.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same',kernel_initializer=layer_init)(x1_)  # 128 x 128 x 128

    #x_relu = L.ReLU()(x2_)
    def relu_end(x):
        r = B.relu(x,max_value=1)
        return r
    x_relu = L.Lambda(relu_end)(x2_)
    return x_relu

def convnext_block(x_in,f):
    x = L.Conv2D(filters=f, kernel_size=(7, 7), strides=1, padding='same')(x_in)  # 512 x 512 x 32
    x = L.LayerNormalization()(x)
    x = L.Conv2D(filters=f*4, kernel_size=(1, 1), strides=1, padding='same')(x)  # 512 x 512 x 32
    x = L.LeakyReLU(0.1)(x)
    x = L.Conv2D(filters=f, kernel_size=(1, 1), strides=1, padding='same')(x)  # 512 x 512 x 32
    x = L.Add()([x_in,x])
    return x

def organoid_net_2d_alternate(input):    #in : 512 x 512 x 32

    x0 = L.Conv2D(filters=32, kernel_size=(7, 7), strides=1, padding='same')(input)  # 512 x 512 x 32
    x0 = convnext_block(x0,32)
    x0 = L.Dropout(0.05)(x0)

    x1 = L.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='same')(x0) #256 x 256 x 64
    x1 = convnext_block(x1,64)
    x1 = L.Dropout(0.05)(x1)

    x2 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding='same')(x1) #128 x 128 x 128
    x2 = convnext_block(x2,128)
    x2 = L.Dropout(0.05)(x2)

    x3 = L.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='same')(x2) #64 x 64 x 256
    x3 = convnext_block(x3,256)
    x3 = L.Dropout(0.05)(x3)

    x4 = L.Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding='same')(x3) #32 x 32 x 512
    x4 = convnext_block(x4,512)
    x4 = convnext_block(x4,512)

    x4 = L.Dropout(0.05)(x4)

    xu = L.UpSampling2D(size=(2,2))(x4)                                           #64 x 64 x 256
    xc = L.Concatenate(axis=-1)([x3,xu])
    x3_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(xc)
    x3_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x3_)
    x3_ = L.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x3_)
    x3_ = L.LeakyReLU(0.1)(x3_)

    xu2 = L.UpSampling2D(size=(2,2))(x3_)  # 128 x 128 x 128
    xc2 = L.Concatenate(axis=-1)([x2,xu2])
    x2_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(xc2)
    x2_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x2_)
    x2_ = L.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x2_)  # 128 x 128 x 128
    x2_ = L.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same')(x2_)  # 128 x 128 x 64
#    x2_ = L.LeakyReLU(0.1)(x2_)

    #xu3 = L.UpSampling2D(size=(2,2))(x2_)  # 256 x 256 x 64
    #xc3 = L.Concatenate(axis=-1)([x1,xu3])
    #x1_ = L.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(xc3)
    #x1_ = L.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x1_)
    #x1_ = L.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same')(x1_)  # 128 x 128 x 128

    #x_relu = L.ReLU()(x2_)
    def relu_end(x):
        r = B.relu(x,max_value=1)
        return r
    x_relu = L.Lambda(relu_end)(x2_)
    return x_relu
def organoid_net_v2(imglist,seq):
    img_f_list = []
    img_f_05_list = []
    img_f_025_list = []
    for i in imglist:
        img_f1,img_f05,img_f025 = focusnet(i)
        img_f_list.append(img_f1)
        img_f_05_list.append(img_f05)
        img_f_025_list.append(img_f025)

    input_tensor = np.zeros((512,512,128))
    for es, s in enumerate(seq):
        img_np = img_f_list[es].numpy()
        input_tensor[:,:,s] = img_np
    print(input_tensor) # 이거 일단 설계는 했는데 될까 ...
    input_f = B.constant(input_tensor)
    out_voxel = organoid_net_2d(input_f)
    return img_f_list,img_f_05_list,img_f_025_list,out_voxel

#3d net is not considered yet : 22/01/02
def organoid_net_3d(input):
    f_init = 16
    x1 = L.Conv3D(filters = f_init,kernel_size=(3,3,3),strides=1,padding='same')(input) #256 x 256 x 128 x 32
    x1 = L.BatchNormalization()(x1)
    x1 = L.ReLU()(x1)
    x1 = L.Conv3D(filters = f_init,kernel_size=(3,3,3),strides=1,padding='same')(x1)
    x1 = L.BatchNormalization()(x1)
    x1 = L.ReLU()(x1)

    x2 = L.Conv3D(filters = f_init*2,kernel_size=(3,3,3),strides=(2,2,1),padding='same')(x1)  #128 x 128 x 128 x 64
    x2 = L.BatchNormalization()(x2)
    x2 = L.ReLU()(x2)
    x2 = L.Conv3D(filters = f_init*2,kernel_size=(3,3,3),strides=1,padding='same')(x2)  #128 x 128 x 128 x 64
    x2 = L.BatchNormalization()(x2)
    x2 = L.ReLU()(x2)
    x2 = L.Conv3D(filters = f_init*2,kernel_size=(3,3,3),strides=1,padding='same')(x2)  #128 x 128 x 128 x 64
    x2 = L.BatchNormalization()(x2)
    x2 = L.ReLU()(x2)

    x3 = L.Conv3D(filters = f_init*4,kernel_size=(3,3,3),strides=2,padding='same')(x2)  #64 x 64 x 64 x 128
    x3 = L.BatchNormalization()(x3)
    x3 = L.ReLU()(x3)
    x3 = L.Conv3D(filters = f_init*4,kernel_size=(3,3,3),strides=1,padding='same')(x3)
    x3 = L.BatchNormalization()(x3)
    x3 = L.ReLU()(x3)
    x3 = L.Conv3D(filters = f_init*4,kernel_size=(3,3,3),strides=1,padding='same')(x3)
    x3 = L.BatchNormalization()(x3)
    x3 = L.ReLU()(x3)

    x4 = L.Conv3D(filters = f_init*8,kernel_size=(3,3,3),strides=2,padding='same')(x3)  #32 x 32 x 32 x 256
    x4 = L.BatchNormalization()(x4)
    x4 = L.ReLU()(x4)
    x4 = L.Conv3D(filters = f_init*8,kernel_size=(3,3,3),strides=1,padding='same')(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.ReLU()(x4)
    x4 = L.Conv3D(filters = f_init*8,kernel_size=(3,3,3),strides=1,padding='same')(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.ReLU()(x4)
    x4 = L.Conv3D(filters = f_init*8,kernel_size=(3,3,3),strides=1,padding='same')(x4)
    x4 = L.BatchNormalization()(x4)
    x4 = L.ReLU()(x4)

    x3_ = L.UpSampling3D(size=(2,2,2))(x4)
    x3_ = L.Concatenate(axis = -1)([x3_,x3])
    x3_ = L.Conv3D(filters = f_init*4,kernel_size=(3,3,3),strides=1,padding='same')(x3_)  #64x 64x 64x 128
    x3_ = L.BatchNormalization()(x3_)
    x3_ = L.ReLU()(x3_)
    x3_ = L.Conv3D(filters = f_init*4,kernel_size=(3,3,3),strides=1,padding='same')(x3_)  #64x 64x 64x 64
    x3_ = L.BatchNormalization()(x3_)
    x3_ = L.ReLU()(x3_)
    x3_ = L.Conv3D(filters = f_init*4,kernel_size=(3,3,3),strides=1,padding='same')(x3_)  #64x 64x 64x 64
    x3_ = L.BatchNormalization()(x3_)
    x3_ = L.ReLU()(x3_)

    x2_ = L.UpSampling3D(size=(2,2,2))(x3_)
    x2_ = L.Concatenate(axis = -1)([x2_,x2])
    x2_ = L.Conv3D(filters = f_init*2,kernel_size=(3,3,3),strides=1,padding='same')(x2_)  #64x 64x 64x 128
    x2_ = L.BatchNormalization()(x2_)
    x2_ = L.ReLU()(x2_)
    x2_ = L.Conv3D(filters = f_init*2,kernel_size=(3,3,3),strides=1,padding='same')(x2_)  #64x 64x 64x 64
    x2_ = L.BatchNormalization()(x2_)
    x2_ = L.ReLU()(x2_)
    x2_ = L.Conv3D(filters = f_init*2,kernel_size=(3,3,3),strides=1,padding='same')(x2_)  #64x 64x 64x 64
    x2_ = L.BatchNormalization()(x2_)
    x2_ = L.ReLU()(x2_)

    x2_ = L.Conv3D(filters = 3,kernel_size=(3,3,3),strides=1,padding='same',activation='softmax')(x2_)  #64x 64x 64x 1

    def sq(x):
        x_sq = B.squeeze(x, -1)
        return x_sq

    return x2_