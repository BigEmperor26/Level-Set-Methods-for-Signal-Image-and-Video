## level set with redistancing for contour tracking in video
# The idea is to change F at each frame and have both phi and level set evolve to change the shape of the curve. 
# Doesn't work well ...
# computes a level set wit doing redistancing as the signed distance function
# therefore the function is saved
# the F velocity is proportional to the gradient of image
# this way the level set stops at the edges of the image
# 

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
from scipy import ndimage
import skfmm

# path of the video to analyse
path_source = 'images/test images/merging_bubbles_animation.mp4'
# path of the 3D frames
path = 'images/merging bubbles animation/image 3D{}.png'
# path of the 2D frames of contour
path_contour = 'images/merging bubbles animation/image contour{}.png'
# path of the gif
path_gif = 'images/merging bubbles animation/out.gif'
# vertical starting point
v_start = 0
# vertical max value of the 3d plot
v_max = 100
# number of iterations of the level set propagation 
n_iter = 100
# number of frames to be saved
frames = 100
#step how many frames to skip for the animation. 1 keep all frames
step = 1
# delta t of time
dt = 0.05
# velocity of the surface that multiplies F(x,y)
v = 10.0
# every how many iterations perform redistancing
redistancing_iter = 5
# number of iterations of level sets to be run for each frame
n_iter_frame = 1


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
# stopping function for the level set. 
def stopping_fun(x):
    return 1. / (1. + norm(grad(x))**2)

# concatenate the frames to make a gif
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst



# open the video flow
cap = cv.VideoCapture(path_source)
while not cap.isOpened():
    cap = cv.VideoCapture(path_source)
    cv.waitKey(1000)
    # print ("Wait for the header")

pos_frame = cap.get(cv.CAP_PROP_POS_FRAMES)

flag, img = cap.read()
if flag:
    # The frame is ready and already captured
    # cv.imshow('video', img)
    pos_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
else:
    # The next frame is not ready, so we try to read it again
    cap.set(cv.CAP_PROP_POS_FRAMES, pos_frame-1)
    # It is better to wait for a while for the next frame to be ready
    cv.waitKey(1000)

img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# apply gaussian filter to img
img_gauss = ndimage.gaussian_filter(img, sigma=1)
# center on the mean
img_mean = img_gauss - np.mean(img_gauss)

# limits of the mesh, x start, x end, y start, y end 
limits = [-int(img.shape[1]/2),int(img.shape[1]/2),-int(img.shape[0]/2),int(img.shape[0]/2)]
# extend of the heatmap to plot
extent = [-int(img.shape[1]/2),int(img.shape[1]/2),-int(img.shape[0]/2),int(img.shape[0]/2)]
# x = np.asarray([i for i in range(img.shape[0])] )
# y = np.asarray([i for i in range(img.shape[1])] )
x = np.linspace(-int(img.shape[1]/2),int(img.shape[1]/2)-1, img.shape[1])
y = np.linspace(-int(img.shape[0]/2),int(img.shape[0]/2)-1, img.shape[0])
x,y = np.meshgrid(x,y)


# initial shape
# circle in this case
# z = grad(img_mean)
# z = z-np.mean(z)

# phi surface function as distance from level set 0 the the image
phi = redistancing(img_mean)

# F velocity field according to gradient of mean image
F = np.ones(img.shape)
F = stopping_fun(img_mean) * v


#frame count
frame_draw = [i*n_iter/frames for i in range(frames+1)]
fr_count = 0

for i in range(n_iter):
    
    # read the next frame
    flag, img = cap.read()
    if flag:
        # The frame is ready and already captured
        # cv.imshow('video', img)
        pos_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv.imreadCAP_PROP_POS_FRAMES, pos_frame-1)
        # It is better to wait for a while for the next frame to be ready
        cv.waitKey(1000)
    if cap.get(cv.CAP_PROP_POS_FRAMES) == cap.get(cv.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # apply gaussian filter to img
    img_gauss = ndimage.gaussian_filter(img, sigma=1)
    # center on the mean
    img_mean = img_gauss - np.mean(img_gauss)

    # F velocity field according to gradient of mean image
    F = np.ones(img.shape)
    F = stopping_fun(img_mean) * v
    
    # for each frame, compute a number of iterations of the level set
    for n in range(n_iter_frame):
        if (i+n)%redistancing_iter == 0:
            phi = redistancing(phi)
        # gradient
        dphi = grad(phi)
        # norm of gradient
        dphi_norm = norm(dphi)
        phi = phi - dt * ( F * dphi_norm)
        # compute the zero level set  
        intersection = np.zeros_like(phi)
        for k in range(x.shape[0]):
            for j in range(y.shape[1]):
                intersection[k,j] = intersect(phi[k,j],0,0.5)
        mask = np.ma.masked_where(intersection == 0, intersection)

    if i==frame_draw[fr_count]:
        # create the 2D plot
        fig, axs = plt.subplots(2,2,sharey=True,sharex=True)
        axs[0,0].imshow(img, extent=extent, cmap='gray')
        axs[0,0].set_title('Original')
        axs[0,1].imshow(F, extent=extent, cmap='gray')
        axs[0,1].set_title('F')
        # axs[0,1].imshow(img_gauss, extent=extent, cmap='gray')
        # axs[0,1].set_title('Blurred')
        axs[1,0].imshow(phi, extent=extent, cmap='gray')
        axs[1,0].set_title('Phi')
        axs[1,1].imshow(intersection, extent=extent, cmap='gray')
        axs[1,1].set_title('contour')
        # plt.show()
        plt.savefig(path_contour.format(fr_count))
        plt.close()
        # create the 3d plot
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        title = ax.set_title('3D Phi')
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
    
# concatenates the frames and make a gif        
im_frame = []
for c in range(0,frames,step):
    im1 = Image.open(path.format(c))
    # im1 = im1.resize((1120, 840),)
    im2 = Image.open(path_contour.format(c))
    # im2 = im2.resize((1120, 840),)
    im3 = get_concat_v(im2, im1)
    im_frame.append (im3)

im_frame[0].save(path_gif, save_all=True, append_images=im_frame[1:], duration=100, loop=0)