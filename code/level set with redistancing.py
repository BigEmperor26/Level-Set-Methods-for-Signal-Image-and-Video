## level set wit redistancing
# computes a level set wit doing redistancing as the signed distance function
# 
# therefore the function is saved 


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
from scipy import ndimage
import skfmm

# path of the frames
path = 'images/level set redistancing/image{}.png'
# path of the gif
path_gif = 'images/level set redistancing/out.gif'
# limits of the mesh, x start, x end, y start, y end 
limits = [-10,10,-10,10]
# extend of the heatmap to plot
extent = [-10,10,-10,10]
# resolution of the meshgrid in both sizes
resolution = 100
# vertical starting point
v_start = 0
# vertical max value
v_max = 600
# size of the mesh
size = 1000
# number of iterations
n_iter = 200
# number of frames
frames = 100
#step how many frames to skip for the animation
step = 1
# delta t of time
dt = 0.05
# velocity of the surface F(x,y)
v = 50.0
# every how many iterations perform redistancing
redistancing_iter = 10

def grad(x):
    grad = np.array(np.gradient(x,edge_order=2))
    return grad

def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))

def redistancing(phi):
    sd = skfmm.distance(phi, dx = 1)
    return sd

def intersect(z,c,eps):
    if z <= c+eps and z >=c-eps:
        return 1
    else:
        return 0  

# create the matrix
x = np.linspace(limits[0], limits[1], size)
y = np.linspace(limits[2], limits[3], size)
# meshgrid
x,y = np.meshgrid(x,y)
# apply the function
z = np.sqrt(x**2+y**2)-1

# initial phi surface function
phi = np.sqrt(x**2+y**2)-1
# F velocity filed
F = np.ones(z.shape)*v

#frame count
frame_draw = [i*n_iter/frames for i in range(frames+1)]
fr_count = 0
for i in range(n_iter):
    if i%redistancing_iter == 0:
        phi = redistancing(phi)
    # gradient
    dphi = grad(phi)
    # norm of gradient
    dphi_norm = norm(dphi)
    phi = phi - dt * ( F * dphi_norm)
    # compute the zero level set  
    intersection = np.zeros_like(phi)
    for k,x_i in enumerate(x):
        for j,y_j in enumerate(y):
            intersection[k,j] = intersect(phi[k,j],0,0.5)
    mask = np.ma.masked_where(intersection == 0, intersection)

    if i==frame_draw[fr_count]:
        # create the 3d plot
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        title = ax.set_title('3D Test')
        ax.set_zlim(v_start, v_max)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.plot_surface(x,y,phi,cmap='viridis', edgecolor='none',alpha=0.5)
        ax.scatter(x,y,mask,cmap='cool', edgecolor='none')
        # plt.show()
        plt.savefig(path.format(fr_count))
        plt.close()
        print(fr_count)
        fr_count+=1
im_frame = []
for c in range(0,frames,step):
    im1 = Image.open(path.format(c))    
    im_frame.append (im1)

im_frame[0].save(path_gif, save_all=True, append_images=im_frame[1:], duration=1, loop=0)