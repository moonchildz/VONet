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
from pygmsh.occ.geometry import Geometry as geom
import meshgen as mg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d
import matplotlib.colors as clr
#import draw_organoid as do
import nucleus_gen as ng
import gc
pi = 3.141592
img_size = 1024
mesh_route = 'D:/mesh_factory/test_3dp.stl'
mesh_route_post = 'D:/mesh_factory/test_3dp_post.stl'
nuc = ng.gen_nuc()
y,x = np.nonzero(nuc)
max_x = np.max(x)
max_y = np.max(y)
min_x = np.min(x)
min_y = np.min(y)


organoid_num = 5000
organoid_count=2283
for i in range(organoid_count,organoid_num):#organoid_num):
    #if i>=1 : continue
    smooth_num = rnd.randint(550, 1999)

    print('Making '+str(organoid_count)+'th Organoid')
    fi = 'shape_'+str(organoid_count)+'.stl'
    fv = 'voxel_'+str(organoid_count)+'.xyz'
    folder_i = 'D:/VO_ver.2/'+str(organoid_count)+'/'
    if os.path.isdir(folder_i) == False:
        os.mkdir(folder_i)

    route_shape = folder_i+fi
    organoid_mesh,success = mg.overall_shape(fi) # as voxel
    if success == False:
        print('Mesh build failed')
        continue
    print('Mesh_loaded')
    density = 1/6.4
    density_z = 1/3.2
    organoid_surf = organoid_mesh.extract_surface()
    organoid_surf = organoid_surf.smooth(n_iter = smooth_num)
    organoid_surf.save(route_shape)
    x_min=-5.0
    x_max=5.0
    y_min=-5.0
    y_max=5.0
    z_min=-5.0
    z_max=5.0
    x = np.arange(x_min, x_max, density)
    y = np.arange(y_min, y_max, density)
    z = np.arange(z_min, z_max, density)
    x, y, z = np.meshgrid(x, y, z)

    print('predicted_grid:', int((x_max - x_min) / density ), int((y_max - y_min) / density ),
          int((z_max - z_min) / density ))
    grid = pv.StructuredGrid(y, x, z)
    ugrid = pv.UnstructuredGrid(grid)
    selection = ugrid.select_enclosed_points(organoid_surf,tolerance=0.0,check_surface=False)
    mask = selection.point_arrays['SelectedPoints'].view(bool)
    mask = mask.reshape(z.shape)  # -> Voxel data_ all_filled
    print('Voxelized:',mask.shape)
    xx,yy,zz = np.nonzero(mask)
    maxz = np.max(zz)
    minz = np.min(zz)

#    ax = plt.axes(projection='3d')
#    ax.scatter(xx, yy, zz, c=zz, cmap='viridis', linewidth=0.5);
#    ax.set_title('surface')
#    ax.set_xlabel('x')
#    ax.set_ylabel('y')
#    ax.set_zlabel('z')
#    plt.show()
    v_p, v_dx,v_dy,v_dz,v_zr = mg.make_cell_point(mask, density)

    list_xyzr = mg.save_vox_org(folder_i, fv, v_p,v_dx,v_dy,v_dz,v_zr)
    print (len(list_xyzr))
    voxel = np.zeros((512,512,64))
    voxel_focus = np.zeros((512,512,64))
    voxel_max_f_only = np.zeros((512,512,64))
    if os.path.isdir(folder_i+'img_focus/') == False:
        os.mkdir(folder_i+'img_focus/')
    if os.path.isdir(folder_i+'img_max_focus/') == False:
        os.mkdir(folder_i+'img_max_focus/')

    list_xyzr = list_xyzr[::-1]
    for i, xyzr in enumerate(list_xyzr):
#        print(xyzr)
        ng.draw_nuke(voxel,voxel_focus,voxel_max_f_only,xyzr[0],xyzr[1],xyzr[2],xyzr[3],maxz,minz)
        #voxel_dot[xyzr[0],xyzr[1],xyzr[2]] = 1.0
        if i%2000==0:
            print('Processing',i,'th cell')
    if os.path.isdir(folder_i+'img/') == False:
        os.mkdir(folder_i+'img/')
    for z_lev in range(128):
        img = voxel[:,:,z_lev]
        img_foc = voxel_focus[:,:,z_lev]
        img_foc_mx = voxel_max_f_only[:,:,z_lev]

        cv2.imwrite(folder_i+'img/'+str(z_lev)+'.png',img*255)
        cv2.imwrite(folder_i+'img_focus/'+str(z_lev)+'.png',img_foc*255)
        cv2.imwrite(folder_i + 'img_max_focus/' + str(z_lev) + '.png', img_foc_mx * 255)

      #  cv2.imshow('Or',img)
      #  cv2.waitKey(50)
#    print (list_xyzr)
#    quit()
    vp = open(folder_i+fv,'r').readlines()[1:]
    print('Voxel making completed.')

#    confocal_img_list = mg.draw_confocal_ellipses(organoid_count,vx,vc, vr1, vr2,vr3, vr4, va, vdx, vdy, vdz, vi,density)
    #confocal_img_list = mg.draw_confocal_ellipses2(organoid_count,vp, density)
    organoid_count+=1
    gc.collect()
#p = pv.Plotter()
#p.add_mesh(organoid_surf, opacity=0.7, line_width=1,show_scalar_bar=True)
#p.add_mesh(surf_cell, opacity=1.0, line_width=1,show_scalar_bar=True)
#p.add_mesh(slice_org)
#p.add_mesh(slice_cell)
#pv.plot(grid.points, scalars=mask)
