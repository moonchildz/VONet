import matplotlib.pyplot as plt
import numpy as np
import cv2
import random as rnd
import perlin_gen as pn

n1 = pn.generate_perlin_noise_2d((1024,1024),(32,32))
n1_l = pn.generate_perlin_noise_2d((1024,1024),(32,32))
n2 = pn.generate_fractal_noise_2d((1024,1024),(64,64),4)
n1 = (n1-np.min(n1))/2
n1_l = (n1_l-np.min(n1_l))/2
n2 = (n2-np.min(n2))/2
k_er = np.ones((3,3))

#np.random.seed(0)
def gen_nuc():

    img = np.zeros((128,128))
    img_line = np.zeros((128,128))
    axis1 = rnd.randint(35,42)
    axis2 = rnd.randint(axis1,42)
    angle = rnd.randint(0,360)
    delta = rnd.randint(40,90)
    pt1 = np.array(cv2.ellipse2Poly((64, 64), (axis1, axis2), angle, 0, 360, delta))
    shuffle_radius = ((np.random.rand(pt1.shape[0])*0.7+0.3)*20)
    shuffle_angle = np.random.rand(pt1.shape[0])*3.141592*2
    shuffle_x = shuffle_radius*np.cos(shuffle_angle)
    shuffle_y = shuffle_radius*np.sin(shuffle_angle)
    sh_rad = np.concatenate([np.expand_dims(shuffle_y,-1),np.expand_dims(shuffle_x,-1)],-1).astype(np.int32)
   # shuffle_rad = ((np.random.rand(pt1.shape[0],pt1.shape[1])-0.5)*40).astype(np.int32)    # -20 ~ 20
#    shuffle_rad = (np.random.uniform(low = 10.0, high = 27.0,size=(pt1.shape))).astype(np.int32)    # value should not exceed min value of axis
#    signal = rnd.sample([-1,1],1)

    pt_shuffled = pt1+sh_rad
    #print(shuffle_rad)

    cv2.fillPoly(img, [pt_shuffled], (1.0))
    #cv2.polylines(img, [pt_shuffled], True, (1.0), 3)
    #cv2.polylines(img_line, [pt_shuffled], True, (1.0), 3)
    img = cv2.GaussianBlur(img,(91,91),0)
    img_line[img<0.75] = 1.0
    img_line[img<0.7] = 0
    x_rndcrop = rnd.randint(65,1024-65)
    y_rndcrop = rnd.randint(65,1024-65)
    n1_crop = n1[y_rndcrop-64:y_rndcrop+64,x_rndcrop-64:x_rndcrop+64]
    n2_crop = n2[y_rndcrop-64:y_rndcrop+64,x_rndcrop-64:x_rndcrop+64]
    n1_l_crop = n1_l[y_rndcrop-64:y_rndcrop+64,x_rndcrop-64:x_rndcrop+64]
    img_line = cv2.GaussianBlur(img_line*n1_l_crop*n1_l_crop,(11,11),0)#cv2.GaussianBlur(img_line,(11,11),0)*n2
    img[img<0.7] = 0

    img_cell = (img*n1_crop*n2_crop)#* noise_snp
    img_cell=cv2.GaussianBlur(img_cell,(5,5),0)
    img_cell = img_cell + img_line
    img_cell[img_cell>1.0] = 1.0
    #pt = np.array(np.random.rand(10,2)*236+10,np.int32)
    #img = cv2.fillConvexPoly(img, pt, (255))
    # 계획 : ellipse2Poly -> Random points -> random +- -> 냅다 fillpoly -> Gaussian -> Profit
    img_cell = cv2.resize(img_cell,None,fx=64/128,fy=64/128)
    return img_cell

cv2.imshow('gen',gen_nuc())
cv2.waitKey()
def defocus_img(img,f,dx,dy):
    #print('Focus',f)
    win = int(3/f)*2+1
    img_gau = cv2.GaussianBlur(img,(win,win),0)*f
    return img_gau
def defocus_cell(img,dz):
    #cv2.imshow('bf',img)
    img_er = np.copy(img)
    for i in range(dz):
        #img_er = cv2.erode(img,kernel=ker,iterations=1)
        img_er = cv2.filter2D(img_er,ddepth=-1,kernel=ker,anchor=(-1,-1))
    #cv2.imshow('af',img_er)
    #cv2.waitKey()
    return img_er
def cell_kaburi(vox,img_cell,img_cell_focus,xx,yy,zz,w,h):
    img_cf_er = cv2.erode(img_cell_focus, k_er, iterations=3)
    vox[xx - w // 2: xx + w // 2, yy - h // 2: yy + h // 2, zz][img_cf_er>0] = 0
    vox[xx - w // 2: xx + w // 2, yy - h // 2: yy + h // 2, zz] = np.maximum(vox[xx-w//2:xx+w//2,yy-h//2:yy+h//2,zz],img_cell)
#    return
def draw_nuke(voxel,voxel2,voxel3,x,y,z,r,maxz,minz):
    xx = x
    yy= y
    #minz 라야 최대가 된다
    slope = 1.05/(maxz+16-minz)
    b = -minz/(maxz+16-minz)
    if xx > 1024-32 or xx < 32: return
    if yy > 1024-32 or yy < 32: return

    img_nuke = gen_nuc()#nuke(img_ell=img_temp,minrad=r_//4,maxrad=r_,color=(c))
    for zz in range(z-r,z+r+1):
        if zz>127 or zz<0 : continue
        dz = abs(zz-z)
        #zz : Actual z value now watching , z : Center of a nue
#        inten = (r-dz+rnd.uniform(0,0.2))/(r)  #dz 따라 달라질 color 정도 -> 이걸 dz 따라가는 rate 개념으로 해야겠다 도저히 못참겠군

        inten = max(1.6-zz*slope-b+rnd.uniform(-0.1,0.1),0) #0.8*(128-z)/128*(r-dz)/(r)+rnd.uniform(-0.1,0.1)  #dz 따라 달라질 color 정도 -> 이게 씨발 왜 1차함수야 ㅋㅋㅋㅋㅋ 병신ㅈ망 Optics
        #print(inten)
        img_er = np.copy(img_nuke)
        img_foc = np.zeros_like(img_nuke)
        img_er = cv2.GaussianBlur(img_er,(3,3),0)
        if dz !=0:
            img_er = cv2.erode(img_er,k_er,iterations= dz)
        for g in range(dz):
            img_er = cv2.GaussianBlur(img_er,(g*2+1,g*2+1),0)*0.9
        img_foc[img_er>0.03] = 1.0-dz*0.15
        img_foc = cv2.GaussianBlur(img_foc,(5,5),0)
        img_foc[img_foc>0.2] = 1.0-dz*0.15

#        cv2.imshow('Er',img_er)
#        cv2.imshow('Foc',img_foc)
#        cv2.moveWindow('Er',0,0)
#        cv2.moveWindow('Foc',200,0)
#        cv2.waitKey()
        w = img_er.shape[1]
        h = img_er.shape[0]

#        print(w,h)
    # print('shape',voxel[x-24:x+24,y-24:y+24,zz].shape)
#        cell_kaburi(voxel,img_er*inten,img_foc,xx,yy,zz,w,h)
        voxel[xx-w//2:xx+w//2,yy-h//2:yy+h//2,zz]=np.maximum(voxel[xx-w//2:xx+w//2,yy-h//2:yy+h//2,zz],img_er*inten)
        voxel2[xx-w//2:xx+w//2,yy-h//2:yy+h//2,zz]=np.maximum(voxel2[xx-w//2:xx+w//2,yy-h//2:yy+h//2,zz],img_foc)
        if dz == 0:
            voxel3[xx - w // 2:xx + w // 2, yy - h // 2:yy + h // 2, zz] = np.maximum(
                voxel3[xx - w // 2:xx + w // 2, yy - h // 2:yy + h // 2, zz], img_foc)
