import os
import sys
import numpy as np
import cv2
import trimesh as tm
import math
import random as rnd
import pygmsh as pgm
import pyvista as pv
import meshio as mio
import multiprocessing as mulproc
import scipy
import time
from pygmsh.occ.geometry import Geometry as geom
pi = 3.141592
img_size = 1024
cell_prob = 1/22
prob_x = 0.75
prob_y = 0.75
prob_z = 0.5

cell_max_size = 64

def make_ellipsoid(geo,x,y,z,r1,r2,r3,a1,a2,a3):
    e = geo.add_ellipsoid((x,y,z), radii=(r1,r2,r3))
    geo.rotate(e,point=(x,y,z),angle = a3,axis=(0.0,0.0,1.0))
    geo.rotate(e,point=(x,y,z),angle = a2,axis=(0.0,1.0,0.0))
    geo.rotate(e,point=(x,y,z),angle = a1,axis=(1.0,0.0,0.0))
    return e
def save_vox_org(foldername,fv,vxp,vx_dx,vx_dy,vx_dz,vx_zr):
    route_vx = foldername+fv
    file_vx = open(route_vx,'w')
    file_vx.write(str(vxp.shape[0])+' '+str(vxp.shape[1])+' '+str(vxp.shape[2])+'\n')
    list_xyzr = []

    for z in range(vxp.shape[0]):
        for y in range(vxp.shape[1]):
            for x in range(vxp.shape[2]):
                if vxp[x, y, z] != 0:
                    file_vx.write(str(vxp[x,y,z]) + ' ' + str(x) + ' ' + str(y) + ' ' + str(z)+'\n')
                    dx = vx_dx[x,y,z]
                    dy = vx_dy[x,y,z]
                    dz = vx_dz[x,y,z]
                    zr = vx_zr[x,y,z]

                    list_xyzr.append((x*16+dx,y*16+dy,z*2+dz,zr))
    return list_xyzr
def make_cell_point(voxel,density): #통상적으로 density = 1/6.4 = 0.15625
    #print(voxel.shape)
    voxcell = np.zeros_like(voxel,dtype=np.int)
    voxdeltax = np.zeros_like(voxel,dtype=np.int)
    voxdeltay = np.zeros_like(voxel,dtype=np.int)
    voxdeltaz = np.zeros_like(voxel,dtype=np.int)
    voxzrad = np.zeros_like(voxel,dtype=np.int)

    voxintensity = np.zeros_like(voxel,dtype=np.float)

    count = 0
    for x_int in range(int(10 / density)):
       # coin_x = rnd.uniform(0,1)
       # if coin_x >=prob_x:continue
        for y_int in range(int(10 / density)):
        #    coin_y = rnd.uniform(0,1)
           # if coin_y >=prob_y:continue
            for z_int in range(int(10 / density)):
         #       coin_z = rnd.uniform(0,1)
                #if coin_z >=prob_z:continue
                cointoss = rnd.uniform(0, 1)
                if cointoss>0.2613:continue
                #if z_int%4 !=2:continue
                if voxel[x_int, y_int, z_int]==True:
                    if voxcell[x_int, y_int, z_int-1] == 1:continue
                    if voxcell[x_int, y_int, z_int-2] == 1:continue
                    count+=1
                    voxcell[x_int, y_int, z_int] = 1
                    voxdeltax[x_int, y_int, z_int] = rnd.randint(-1*int(cell_max_size*0.25),int(cell_max_size*0.25))  #Integer
                    voxdeltay[x_int, y_int, z_int] = rnd.randint(-1*int(cell_max_size*0.25),int(cell_max_size*0.25))  #Integer
                    voxdeltaz[x_int, y_int, z_int] = rnd.randint(-1,1)  #Integer
                    voxzrad[x_int, y_int, z_int] = rnd.randint(3,6)  #Integer
    return voxcell, voxdeltax,voxdeltay,voxdeltaz,voxzrad
def find_max_z(voxel):
    min_z = 0
    max_z = 0
    vz=[]
    for v in voxel:
        vv = int(v.split(' ')[3])
        vz.append(vv)
    vz=np.array(vz)
    min_z=np.min(vz)
    max_z = np.max(vz)
    return min_z,max_z
def make_nefron(num,nummax):
    neflist= []
    with pgm.occ.Geometry() as geo2:

        th_big = num * pi/nummax/2 + rnd.uniform(-pi/45, pi/45) - pi/4
        print(th_big)
        ps_big = rnd.uniform(-pi/4, pi/4)


        geo2.characteristic_length_max = 0.05
        x = 1
        y = 0
        z = 0

        r1 = rnd.uniform(0.3, 0.8)
        r2 = rnd.uniform(r1 * 0.8, r1 * 1.2)
        r3 = rnd.uniform(r1 * 0.8, r1 * 1.2)
        th = rnd.uniform(-pi, pi)
        ps = rnd.uniform(-pi, pi)
        py = rnd.uniform(-pi, pi)
        e = geo2.add_ellipsoid((x, y, z), radii=(r1, r2, r3))
        geo2.rotate(e, point=(x, y, z), angle=py, axis=(0.0, 0.0, 1.0))
        geo2.rotate(e, point=(x, y, z), angle=ps, axis=(0.0, 1.0, 0.0))
        geo2.rotate(e, point=(x, y, z), angle=th, axis=(1.0, 0.0, 0.0))

        geo2.rotate(e, point=(-2,0,0), angle=th_big, axis=(0.0, 1.0, 0.0))
        #geo.rotate(e, point=(0,0,0), angle=ps_big, axis=(0.0, 1.0, 0.0))

        neflist.append(e)

        x_ = x-rnd.uniform(0.2,0.3)
        r1 = rnd.uniform(r1*0.5, r1*1.3)
        r2 = rnd.uniform(r1 * 0.8, r1 * 1.2)
        r3 = rnd.uniform(r1 * 0.8, r1 * 1.2)
        th = rnd.uniform(-pi, pi)
        ps = rnd.uniform(-pi, pi)
        py = rnd.uniform(-pi, pi)
        e2 = geo2.add_ellipsoid((x_, y, z), radii=(r1, r2, r3))
        geo2.rotate(e2, point=(x_, y, z), angle=py, axis=(0.0, 0.0, 1.0))
        geo2.rotate(e2, point=(x_, y, z), angle=ps, axis=(0.0, 1.0, 0.0))
        geo2.rotate(e2, point=(x_, y, z), angle=th, axis=(1.0, 0.0, 0.0))

        geo2.rotate(e2, point=(-2,0,0), angle=th_big, axis=(0.0, 1.0, 0.0))
        #geo.rotate(e, point=(0,0,0), angle=ps_big, axis=(0.0, 1.0, 0.0))

        neflist.append(e2)

        y_ = y+rnd.uniform(-0.2,0.2)
        r1 = rnd.uniform(r1*0.4, r1*0.6)
        r2 = rnd.uniform(r1 * 0.8, r1 * 1.2)
        r3 = rnd.uniform(r1 * 0.8, r1 * 1.2)
        th = rnd.uniform(-pi, pi)
        ps = rnd.uniform(-pi, pi)
        py = rnd.uniform(-pi, pi)
        e3 = geo2.add_ellipsoid((x, y_, z), radii=(r1, r2, r3))
        geo2.rotate(e3, point=(x, y_, z), angle=py, axis=(0.0, 0.0, 1.0))
        geo2.rotate(e3, point=(x, y_, z), angle=ps, axis=(0.0, 1.0, 0.0))
        geo2.rotate(e3, point=(x, y_, z), angle=th, axis=(1.0, 0.0, 0.0))

        geo2.rotate(e3, point=(-2,0,0), angle=th_big, axis=(0.0, 1.0, 0.0))
        #geo.rotate(e, point=(0,0,0), angle=ps_big, axis=(0.0, 1.0, 0.0))

        neflist.append(e3)


        mesh = geo2.generate_mesh()
        mesh.write('d:/mesh_factory/nefron.stl')
        meshdata = pv.read('d:/mesh_factory/nefron.stl')
    return meshdata

def overall_shape(fname,is_kidney):
    fname_temp = 'D:/mesh_factory/'+fname
    print(fname_temp)
    balllist = []
    cx = 0
    cy = 0
    cz = 0
    size_cyst = 0
    boundary = None
    param_list= []
    with pgm.occ.Geometry() as geo:

        geo.characteristic_length_max = 0.1
        suc = False
        x = 1.0
        y = 0
        z = -1.0

        r1 = rnd.uniform(1.6, 2.0)
        r2 = rnd.uniform(r1 * 0.8, r1 * 1.2)
        r3 = rnd.uniform(r1 * 0.8, r1 * 1.2)
        size_cyst = r1

        th = rnd.uniform(-pi, pi)
        ps = rnd.uniform(-pi, pi)
        py = rnd.uniform(-pi, pi)
        e = geo.add_ellipsoid((x, y, z), radii=(r1, r2, r3))
        geo.rotate(e, point=(x, y, z), angle=py, axis=(0.0, 0.0, 1.0))
        geo.rotate(e, point=(x, y, z), angle=ps, axis=(0.0, 1.0, 0.0))
        geo.rotate(e, point=(x, y, z), angle=th, axis=(1.0, 0.0, 0.0))
        param_list.append([x,y,z,r1,r2,r3,th,ps,py])
        balllist.append(e)

        x2 = rnd.uniform(x+r1*0.1, x+r1*0.2)
        y2 = rnd.uniform(y+r2*0.1, y+r2*0.2)
        z2 = rnd.uniform(z+r3*0.5, z+r3*0.8)

        r1_ = rnd.uniform(1.6, 2.0)
        r2_ = rnd.uniform(r1_ * 0.8, r1_ * 1.2)
        r3_ = rnd.uniform(r1_ * 0.8, r1_ * 1.2)

        th = rnd.uniform(-pi, pi)
        ps = rnd.uniform(-pi, pi)
        py = rnd.uniform(-pi, pi)
        e2 = geo.add_ellipsoid((x2, y2, z2), radii=(r1_, r2_, r3_))
        print()
        geo.rotate(e2, point=(x2, y2, z2), angle=py, axis=(0.0, 0.0, 1.0))
        geo.rotate(e2, point=(x2, y2, z2), angle=ps, axis=(0.0, 1.0, 0.0))
        geo.rotate(e2, point=(x2, y2, z2), angle=th, axis=(1.0, 0.0, 0.0))
        param_list.append([x2,y2,z2,r1_,r2_,r3_,th,ps,py])
        balllist.append(e2)
        x3 = x / 2 + x2 / 2 + rnd.uniform(-0.2, 0.2)
        y3 = y / 2 + y2 / 2 + rnd.uniform(-0.2, 0.2)
        z3 = z / 2 + z2 / 2 + rnd.uniform(-0.2, 0.2)

        cointoss = rnd.uniform(0,1)
        if cointoss>0.3:
            r1__ = rnd.uniform(r1_*0.6, r1_)
            r2__ = rnd.uniform(r1__ * 0.8, r1__ * 1.2)
            r3__ = rnd.uniform(r1__ * 0.8, r1__ * 1.2)
            th = rnd.uniform(-pi, pi)
            ps = rnd.uniform(-pi, pi)
            py = rnd.uniform(-pi, pi)
            e3 = geo.add_ellipsoid((x3, y3, z3), radii=(r1__, r2__, r3__))
            geo.rotate(e3, point=(x3, y3, z3), angle=py, axis=(0.0, 0.0, 1.0))
            geo.rotate(e3, point=(x3, y3, z3), angle=ps, axis=(0.0, 1.0, 0.0))
            geo.rotate(e3, point=(x3, y3, z3), angle=th, axis=(1.0, 0.0, 0.0))
            param_list.append([x3, y3, z3, r1__, r2__, r3__, th, ps, py])
            balllist.append(e3)
        if is_kidney == False or cointoss >0.7:    # Cyst 돌기 하나 더
            x4 = x3 + rnd.uniform(-0.4, 0.4)
            y4 = y3 + rnd.uniform(-0.4, 0.4)
            z4 = z3 + rnd.uniform(-0.4, 0.4)
            r1__ = rnd.uniform(0.5, 0.8)
            r2__ = rnd.uniform(r1__ * 0.8, r1__ * 1.2)
            r3__ = rnd.uniform(r1__ * 0.8, r1__ * 1.2)
            th = rnd.uniform(-pi, pi)
            ps = rnd.uniform(-pi, pi)
            py = rnd.uniform(-pi, pi)
            e4 = geo.add_ellipsoid((x4, y4, z4), radii=(r1__, r2__, r3__))
            geo.rotate(e4, point=(x4, y4, z4), angle=py, axis=(0.0, 0.0, 1.0))
            geo.rotate(e4, point=(x4, y4, z4), angle=ps, axis=(0.0, 1.0, 0.0))
            geo.rotate(e4, point=(x4, y4, z4), angle=th, axis=(1.0, 0.0, 0.0))
            param_list.append([x4, y4, z4, r1__, r2__, r3__, th, ps, py])

            balllist.append(e4)
        geo.boolean_union(balllist)
        centerpoint = [x3,y3,z3]
        try:
            print('generating mesh')
            mesh = geo.generate_mesh()
            print(mesh)
            mesh.write(fname_temp)
            md = pv.read(fname_temp)
            boundary = md.bounds
            if mesh!=None:
                suc = True
        except:
            pass
        print('Was it Successful ? :',suc)
    if suc == False: return 0, suc,[]
    if is_kidney ==True:
        shell=[]
        with pgm.occ.Geometry() as geo_edge:
            geo_edge.characteristic_length_max = 0.1
            for i in range(len(param_list)):
                x = param_list[i][0] + rnd.uniform(-0.3,0.3)
                y = param_list[i][1]+ rnd.uniform(-0.3,0.3)
                z = param_list[i][2]+ rnd.uniform(-0.3,0.3)
                r1 = param_list[i][3] * rnd.uniform(1.0,1.35)
                r2 = param_list[i][4] * rnd.uniform(1.0,1.35)
                r3 = param_list[i][5] * rnd.uniform(1.0,1.35)
                th = param_list[i][6] + rnd.uniform(-0.2,0.2)
                ps = param_list[i][7] + rnd.uniform(-0.2,0.2)
                py = param_list[i][8] + rnd.uniform(-0.2,0.2)
               # print(param_list,x,y,z,r1,r2,r3,th,ps,py)
                e = geo_edge.add_ellipsoid((x, y, z), radii=(r1, r2, r3))
                geo_edge.rotate(e, point=(x, y, z), angle=py, axis=(0.0, 0.0, 1.0))
                geo_edge.rotate(e, point=(x, y, z), angle=ps, axis=(0.0, 1.0, 0.0))
                geo_edge.rotate(e, point=(x, y, z), angle=th, axis=(1.0, 0.0, 0.0))
                shell.append(e)
            geo_edge.boolean_union(shell)
            try:
                print('generating mesh')
                mesh = geo_edge.generate_mesh()
                mesh.write('D:/mesh_factory/shell.stl')
            except:pass
    if is_kidney == False:
        coin_tumor = rnd.uniform(0,1)
#        boundary = mesh.bounds
        xmin = boundary[0]
        xmax = boundary[1]
        dx = xmax-xmin
        mx = xmax/2+xmin/2
        ymin = boundary[2]
        ymax = boundary[3]
        dy = ymax-ymin
        my = ymax/2+ymin/2
        zmin = boundary[4]
        zmax = boundary[5]
        dz = zmax-zmin
        mz = zmax/2+zmin/2

        with pgm.occ.Geometry() as geo2:

            geo2.characteristic_length_max = 0.1

            calciumlist = []
            num_cal = rnd.randint(0, 8)
            print('numcal', num_cal)
            if num_cal >0:
                for i in range(num_cal):
                    r_cx = rnd.uniform(mx-dx/2.3, mx+dx/2.3)
                    r_cy = rnd.uniform(my-dy/2.3, my+dy/2.3)
                    r_cz = rnd.uniform(mz-dz/2.3, mz+dz/2.3)

                    r1 = rnd.uniform(0.7, 1.8)
                    r2 = rnd.uniform(r1 * 0.5, r1 * 0.9)
                    r3 = rnd.uniform(r1 * 0.02, r1 * 0.4)
                    th = rnd.uniform(-pi, pi)
                    ps = rnd.uniform(-pi, pi)
                    py = rnd.uniform(-pi, pi)
                    ec = geo2.add_ellipsoid((0, 0, 0), radii=(r1, r2, r3))
                    geo2.rotate(ec, point=(0,0,0), angle=ps, axis=(0.0, 1.0, 0.0))
                    geo2.rotate(ec, point=(0,0,0), angle=th, axis=(1.0, 0.0, 0.0))
                    geo2.rotate(ec, point=(0,0,0), angle=py, axis=(0.0, 0.0, 1.0))
                    geo2.translate(ec,[r_cx,r_cy,r_cz])
                    calciumlist.append(ec)

                if num_cal>1:geo2.boolean_union(calciumlist)
                mesh2 = geo2.generate_mesh()
                print('writable?')
                mesh2.write('D:/mesh_factory/calcium.stl')
        if coin_tumor<0.95: # 추후 0.15 정도로 tumor 발생 확률 조정
            tumor_list=[]
            with pgm.occ.Geometry() as geo3:
                geo3.characteristic_length_max = 0.1
                r_cx = rnd.uniform(mx - dx / 2.3, mx + dx / 2.3)
                r_cy = rnd.uniform(my - dy / 2.3, my + dy / 2.3)
                r_cz = rnd.uniform(mz - dz / 2.3, mz + dz / 2.3)

                r1 = rnd.uniform(0.3, 0.7)
                r2 = rnd.uniform(r1 * 0.5, r1 * 0.9)
                r3 = rnd.uniform(r1 * 0.5, r1 * 0.9)
                th = rnd.uniform(-pi, pi)
                ps = rnd.uniform(-pi, pi)
                py = rnd.uniform(-pi, pi)
                ec = geo3.add_ellipsoid((0, 0, 0), radii=(r1, r2, r3))
                geo3.rotate(ec, point=(0, 0, 0), angle=ps, axis=(0.0, 1.0, 0.0))
                geo3.rotate(ec, point=(0, 0, 0), angle=th, axis=(1.0, 0.0, 0.0))
                geo3.rotate(ec, point=(0, 0, 0), angle=py, axis=(0.0, 0.0, 1.0))
                geo3.translate(ec, [r_cx, r_cy, r_cz])
                tumor_list.append(ec)
                for i in range(rnd.randint(2,5)):
                    rrx = rnd.uniform(r_cx-0.1,r_cx+0.1)
                    rry = rnd.uniform(r_cy-0.1,r_cy+0.1)
                    rrz = rnd.uniform(r_cz-0.1,r_cz+0.1)
                    r1 = rnd.uniform(0.3, 0.5)
                    r2 = rnd.uniform(r1 * 0.5, r1 * 0.9)
                    r3 = rnd.uniform(r1 * 0.5, r1 * 0.9)
                    th = rnd.uniform(-pi, pi)
                    ps = rnd.uniform(-pi, pi)
                    py = rnd.uniform(-pi, pi)
                    ec_ = geo3.add_ellipsoid((0, 0, 0), radii=(r1, r2, r3))
                    geo3.rotate(ec_, point=(0, 0, 0), angle=ps, axis=(0.0, 1.0, 0.0))
                    geo3.rotate(ec_, point=(0, 0, 0), angle=th, axis=(1.0, 0.0, 0.0))
                    geo3.rotate(ec_, point=(0, 0, 0), angle=py, axis=(0.0, 0.0, 1.0))
                    geo3.translate(ec_, [rrx, rry, rrz])
                    tumor_list.append(ec)
                geo3.boolean_union(tumor_list)
                mesh3 = geo3.generate_mesh()
                mesh3.write('D:/mesh_factory/tumor.stl')

    meshdata = pv.read(fname_temp)
    return meshdata,suc,centerpoint
def overall_void(fname):
    fname_temp = 'D:/mesh_factory/'+fname
    print(fname_temp)
    balllist = []
    cx = 0
    cy = 0
    cz = 0

    with pgm.occ.Geometry() as geo:

        geo.characteristic_length_max = 0.1
        suc = False

        voidlist = []
        x = rnd.uniform(0.7,1.2)
        y = rnd.uniform(-0.2,0.2)
        z = 0.3+rnd.uniform(0,0.3)
        cx+=x
        cy+=y
        cz+=z
        r1 = rnd.uniform(0.8, 1.2)
        r2 = rnd.uniform(r1 * 0.6, r1 * 0.9)
        r3 = rnd.uniform(r1 * 0.6, r1 * 0.9)

        th = rnd.uniform(-pi, pi)
        ps = rnd.uniform(-pi, pi)
        py = rnd.uniform(-pi, pi)
        e3 = geo.add_ellipsoid((x, y, z), radii=(r1, r2, r3))
        geo.rotate(e3, point=(x, y, z), angle=py, axis=(0.0, 0.0, 1.0))
        geo.rotate(e3, point=(x, y, z), angle=ps, axis=(0.0, 1.0, 0.0))
        geo.rotate(e3, point=(x, y, z), angle=th, axis=(1.0, 0.0, 0.0))
        voidlist.append(e3)

        x = rnd.uniform(0.7,1.2)
        y = rnd.uniform(-0.2,0.2)
        z = -0.7 - rnd.uniform(0, 0.3)
        cx+=x
        cy+=y
        cz+=z

        r1 = rnd.uniform(0.8, 1.2)
        r2 = rnd.uniform(r1 * 0.6, r1 * 0.9)
        r3 = rnd.uniform(r1 * 0.6, r1 * 0.9)

        th = rnd.uniform(-pi, pi)
        ps = rnd.uniform(-pi, pi)
        py = rnd.uniform(-pi, pi)
        e4 = geo.add_ellipsoid((x, y, z), radii=(r1, r2, r3))
        geo.rotate(e4, point=(x, y, z), angle=py, axis=(0.0, 0.0, 1.0))
        geo.rotate(e4, point=(x, y, z), angle=ps, axis=(0.0, 1.0, 0.0))
        geo.rotate(e4, point=(x, y, z), angle=th, axis=(1.0, 0.0, 0.0))
        voidlist.append(e4)

        cx/=2+r1
        cy/=2
        cz/=2

        cxx = cx-rnd.uniform(0.8,1.1)
        cyy = cy+rnd.uniform(-0.1,0.1)
        czz =  cz+rnd.uniform(-0.1,0.1)
        r1 = rnd.uniform(1.8, 2.7)
        r2 = rnd.uniform(r1 * 0.15, r1 * 0.45)
        r3 = rnd.uniform(r1 * 0.15, r1 * 0.45)
        th = rnd.uniform(-pi, pi)
        ps = rnd.uniform(-pi, pi)
        py = rnd.uniform(-pi, pi)
        e5 = geo.add_ellipsoid((cxx, cyy, czz), radii=(r1, r2, r3))
            #geo.rotate(e5, point=(cxx, cyy, czz), angle=th, axis=(1.0, 0.0, 0.0))
        voidlist.append(e5)

        cxx =rnd.uniform(-0.8,-0.5)
        cyy = rnd.uniform(-0.1,0.1)
        czz =  rnd.uniform(-0.1,0.1)
        r1 = rnd.uniform(1.5, 2.1)
        r2 = rnd.uniform(r1 * 0.45, r1 * 0.75)
        r3 = rnd.uniform(r1 * 0.45, r1 * 0.75)
        th = rnd.uniform(-pi, pi)
        ps = rnd.uniform(-pi, pi)
        py = rnd.uniform(-pi, pi)
        e6 = geo.add_ellipsoid((cxx, cyy, czz), radii=(r2, r3, r1))
            #geo.rotate(e5, point=(cxx, cyy, czz), angle=th, axis=(1.0, 0.0, 0.0))
        voidlist.append(e6)

        geo.boolean_union(voidlist)
        centerpoint = [cx,cy,cz]
        try:
            print('generating mesh')
            mesh = geo.generate_mesh()
            print(mesh)
            mesh.write(fname_temp)
            if mesh!=None:
                suc = True
        except:
            pass
        print('Was it Successful ? :',suc)
    if suc == False: return 0, suc
    meshdata = pv.read(fname_temp)
    return meshdata,suc,centerpoint

def overall_cortex(fname,center):
    fname_temp = 'D:/mesh_factory/'+fname
    print(fname_temp)
    balllist = []
    cx = center[0]
    cy = center[1]
    cz = center[2]

    r_cx = rnd.uniform(-.5,.5)
    r_cy = rnd.uniform(-.5,.5)
    r_cz = rnd.uniform(-.5,.5)
    with pgm.occ.Geometry() as geo:

        geo.characteristic_length_max = 0.05
        suc = False


        cortexlist = []
        num = rnd.randint(7,13)
        for i in range(num):
            dist = rnd.uniform(0.8, 1.2) * rnd.choice([-1,1]) + rnd.uniform(-0.5, 1.2)
            dist2 = rnd.uniform(0.8, 1.2) * rnd.choice([-1,1])
            dist3 = rnd.uniform(0.8, 1.2) * rnd.choice([-1,1])

            r1 = rnd.uniform(1.2, 1.8)
            r2 = rnd.uniform(r1 * 0.3, r1 * 0.6)
            r3 = rnd.uniform(r1 * 0.2, r1 * 0.4)

            th = rnd.uniform(-pi, pi)
            ps = rnd.uniform(-pi, pi)
            py = rnd.uniform(-pi, pi)
            e3 = geo.add_ellipsoid((cx, cy, cz), radii=(r1, r2, r3))
            geo.rotate(e3, point=(cx, cy, cz), angle=ps, axis=(0.0, 1.0, 0.0))
            geo.rotate(e3, point=(cx, cy, cz), angle=th, axis=(1.0, 0.0, 0.0))
            geo.rotate(e3, point=(cx, cy, cz), angle=py, axis=(0.0, 0.0, 1.0))
            geo.translate(e3,(dist,dist2,dist3))
            cortexlist.append(e3)
        geo.boolean_union(cortexlist)


        try:
            print('generating mesh')
            mesh = geo.generate_mesh()
            print(mesh)
            mesh.write(fname_temp)
            if mesh!=None:
                suc = True
        except:
            pass
        print('Was it Successful ? :',suc)
    if suc == False: return 0, suc
    meshdata = pv.read(fname_temp)
    return meshdata,suc

def back_organ_gen():
    num_organ = rnd.randint(2,5)
    organlist=[]
    for i in range(num_organ):
        name = 'organ_'+str(i)+'.stl'
        balllist=[]
        with pgm.occ.Geometry() as geo:
            geo.characteristic_length_max = 0.05
            cx = rnd.uniform(-3.2,3.2)
            cy = rnd.uniform(-3.2,3.2)
            cz = rnd.uniform(-3.2,3.2)
            print(cx,cy,cz,'wteryu')
            for ii in range(rnd.randint(2,8)):
                ccx = cx+rnd.uniform(-0.8,0.8)
                ccy = cy+rnd.uniform(-0.8,0.8)
                ccz = cz+rnd.uniform(-0.8,0.8)
                r1= rnd.uniform(0.5,4.0)
                r2= rnd.uniform(0.5,4.0)
                r3= rnd.uniform(0.5,4.0)
                ax = rnd.uniform(-pi,pi)
                ay = rnd.uniform(-pi,pi)
                az = rnd.uniform(-pi,pi)
                e = geo.add_ellipsoid((ccx, ccy, ccz), radii=(r1, r2, r3))
                geo.rotate(e, point=(ccx, ccy, ccz), angle=ax, axis=(0.0, 0.0, 1.0))
                geo.rotate(e, point=(ccx, ccy, ccz), angle=ay, axis=(0.0, 1.0, 0.0))
                geo.rotate(e, point=(ccx, ccy, ccz), angle=az, axis=(1.0, 0.0, 0.0))
#                param_list.append([x, y, z, r1, r2, r3, th, ps, py])
                balllist.append(e)
            geo.boolean_union(balllist)
            try:
                print('generating mesh')
                mesh = geo.generate_mesh()
                print(mesh)
                mesh.write('D:/mesh_factory/'+name)
                if mesh != None:
                    suc = True
            except:
                pass
            print('Was it Successful ? :', suc)
        if suc == False: return 0, suc
        meshdata = pv.read('D:/mesh_factory/'+name)
        organlist.append(meshdata)
    return organlist
