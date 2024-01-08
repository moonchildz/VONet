import gc
import random as rnd
import cv2

import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import ImageTk as itk
from PIL import Image
from tkinter import filedialog as fd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d
import matplotlib.colors as clr
import numpy as np
import os
import sys
import test_org_v3 as to


window = tk.Tk()
def outer_only(voxel):

    vox_n = np.zeros_like(voxel)

    for x in range(voxel.shape[0]):
        for y in range(voxel.shape[1]):
            for z in range(voxel.shape[2]):
                if voxel[x,y,z] ==0:continue
                sum_v = np.sum(voxel[x-1:x+2,y-1:y+2,z-1:z+2])
                if sum_v < 0.99*27:
                    vox_n[x,y,z]=1
    return vox_n

def write_voxel(dir, num,voxel):
    save_org = open(dir+'_'+str(num)+'_dl.txt','w')

    x,y,z = np.nonzero(voxel)
    save_org.write(str(len(x))+'\n\n')
    for xx,yy,zz in zip(x,y,z):
        save_org.write(str(xx)+' '+str(yy)+' '+str(zz)+'\n')

class window_tk():
    def __init__(self,main):
        self.main=main
        self.fig = Figure()
        self.canvas_org_2d = tk.Canvas(self.main, bg='white')
        self.canvas_org_3d = FigureCanvasTkAgg(self.fig,master=main)

        self.btn_load = tk.Button(self.main,text = "Load Organoid folder",command = self.load_imgs)
        self.btn_shape = tk.Button(self.main,text = "Estimate Organoid Shape",command = self.derive_shape)
        self.btn_livedead = tk.Button(self.main,text = "Estimate Live/Dead rate",command = self.derive_rate)
        self.init_canvas(self.main)
        self.org_dir = None
        self.imgnum =0
        self.imgseq =0
        self.axis = None
        self.in_text = ttk.Entry(self.main,width=20)
    # self.id_text.config(state='readonly')
        self.in_ = tk.Label(self.main,text = 'Image Number')

        self.in_.place(x=15,y=530)
        self.in_text.place(x=105, y=530)
        self.img_store = []
        self.ld_net = None
        self.shape_net = None
    def init_canvas(self,main):
        main.geometry('1027x552+100+50')
        self.btn_shape.pack(side=tk.BOTTOM)
        self.btn_livedead.pack(side=tk.BOTTOM)
        self.btn_load.pack(side=tk.BOTTOM)
        self.canvas_org_2d.config(width=512, height=512)
        self.canvas_org_2d.pack(side=tk.LEFT)
        self.canvas_org_3d.get_tk_widget().pack(side=tk.RIGHT)
        self.img_route=None
        self.org_imgs=None
        self.dir_img=''
        self.org_shape = None
        self.z_seq = None
    def load_imgs(self):
        gc.collect()
        self.img_store = []
        self.dir_imgs = fd.askdirectory(parent=window,initialdir='D:/neuron segmentation dataset/livedead images/',title = "select Organoid Folder")
        print(self.dir_imgs)
        try:
            self.imgnum = int(self.in_text.get())
        except:
            self.imgnum = 0
            pass
        imglist = os.listdir(self.dir_imgs)

        print('len of data : ',len(imglist))
        for i in range(64):
            if len(imglist)==128:
                ni = i * 2 + 1
            elif len(imglist)==64:
                ni = i
            print(self.dir_imgs+'/'+ imglist[ni])
            img = cv2.imread(self.dir_imgs + '/'+imglist[ni], -1)
            self.img_store.append(img)
        print(len(self.img_store),self.img_store[0].shape)
    def derive_shape(self):
        imgbox = np.zeros((256,256,64))
        n_img = int(self.in_text.get())
        if n_img <1:
            print('Img number not available')
            return
        if len(self.img_store)!=64:
            print("Img not loaded")
            return
        img_seq = []
        step = int(40/(n_img+1)+0.5)
        for i in range(1, n_img+1):
            img_seq.append(i * step +12)
        print('img used :',img_seq)
        for q in img_seq:
            img = self.img_store[q][:,:,0]/255.0
            img = cv2.resize(img,(256,256))
            imgbox[:,:,q] = img
        input_data = np.expand_dims(imgbox,0)
        if self.shape_net ==None:
            self.shape_net = to.make_model('shape')
        whole_shape = to.derive_shape(self.shape_net,input_data)[0]
        whole_shape[whole_shape != 1] = 0
        write_voxel(self.dir_imgs, self.imgnum, whole_shape)
        self.org_shape = outer_only(whole_shape)
        print('Org Shape :', self.org_shape.shape, np.mean(self.org_shape))

        self.display_org()
        self.display_org_3d()
        for it in self.canvas_org_3d.get_tk_widget().find_all():
            self.canvas_org_3d.get_tk_widget().delete()
    def derive_rate(self):
        if len(self.img_store)!=64:
            print("Img not loaded:",len(self.img_store))
            return
        if self.ld_net ==None:
            self.ld_net = to.make_model('ld')

        img_seq = []
        step = int(40/(7)+0.5)
        for i in range(1, 7+1):
            img_seq.append(i * step +12)
            #Reg/input size : 2 x (256 x 256 x 5)
        i1 = []
        i2=[]
        for q in img_seq:
            img = self.img_store[q][:,:,0]/255.0
            img = cv2.resize(img,(256,256))

            i1.append(img)
            img2 = self.img_store[q][:,:,2]/255.0
            img2 = cv2.resize(img2,(256,256))
            i2.append(img2)
        i1 = np.transpose(np.array(i1),[1,2,0])
        i2 = np.transpose(np.array(i2),[1,2,0])
        total_in = [np.expand_dims(i1,0),np.expand_dims(i2,0)]
        ld_rate = to.derive_ldrate(self.ld_net,total_in)[0]

        print('input data:')
        print(img_seq)
        print(np.max(i1),np.min(i1),'/',np.max(i2),np.min(i2),total_in[0].shape,total_in[1].shape)
        print('Live Dead Rate :',ld_rate)
    def display_org(self):
        self.img_dp = cv2.cvtColor(self.img_store[32], cv2.COLOR_BGR2RGB)
        self.img_dp = cv2.resize(self.img_dp, (256, 256))

        imgitk = itk.PhotoImage(Image.fromarray(self.img_dp))
        self.canvas_org_2d.create_image(0,0,anchor = tk.NW,image=imgitk)
        self.canvas_org_2d.image = imgitk

    def display_org_3d(self):
        #self.canvas_org_3d.get_tk_widget().cle
        self.axis = mplot3d.Axes3D(self.fig)
        self.axis.grid(True)
        self.axis.set_zlim3d(0,64)
        self.axis.set_xlim(0,128)
        self.axis.set_ylim(0,128)
        x,y,z = np.nonzero(self.org_shape)
        x_=[]
        y_=[]
        z_ = []
        c_ = []
        for xx,yy,zz in zip(x,y,z):
            if self.org_shape[xx,yy,zz]==1:
                x_.append(xx)
                y_.append(yy)
                z_.append(zz)
                c_.append(self.org_shape[xx,yy,zz])
        print('Number of Cell : ',len(x_), '<-', x.shape)
#        self.axis.scatter(x,y,z,c=z)
        self.axis.scatter(np.array(x_),np.array(y_),np.array(z_),c=np.array(z_))
        self.canvas_org_3d.get_tk_widget().pack()
        self.canvas_org_3d.draw()
app= window_tk(window)
window.mainloop()