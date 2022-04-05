#!/usr/bin/env python
# coding: utf-8

# # level set notebook
# 
# with redistancing for image segmentation-contour detection.
# Computes a level set doing redistancing as the signed distance function. The F velocity is proportional to the gradient of image this. The level set stops at the edges of the image.
# 

# ## Requirements!
# - Python 3.8
# - numpy 1.21.5
# - opencv-python-4.5.5.64
# - scipy-1.8.0
# - skfmm 2022.3.26
# - Pillow 9.0.0
# - matplotlib 3.5.1
# 
# To install these if you don't have run
# ```
# pip install numpy
# pip install python-opencv
# pip install cipy
# pip install scikit-fmm
# pip install Pillow
# pip install matplotlib
# ```
# 
# **Warning.** Colab by default uses python 3.7 which is unsupported in this code

# Import of the basic libraries that are used

# In[65]:


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
from scipy import ndimage
import skfmm


# Parameters to set

# In[66]:



# path of the image to analyse
path_source = '../images/test images/water_coins.jpg'
# path where to save the 3D frames
path = '../images/water coins/image 3D{}.png'
# path where to save the 2D frames of contour
path_contour = '../images/water coins/image contour{}.png'
# path where to save the gif
path_gif = '../images/water coins/out.gif'
# vertical starting point for the 3d plot
v_start = 0
# vertical max value for the 3d plot
v_max = 100
# number of iterations to run the level set propagation
n_iter = 200
# number of frames to be generated
frames = 100
#step how many frames to skip for the animation
step = 1
# delta t of time of the level set propagation
dt = 0.05
# velocity that multiplies F(x,y)
v = 20.0
# how many iterations to wait before performing redistancing
redistancing_iter = 10
# allow redistancing or not
allow_redistancing = True
# vertical or horizontal concatenation. True vertical, False horizontal
vertical_concat = False


# Basic mathematical functions

# In[67]:


# gradient of the image. 
# Calculated using the central difference method using a high degree of approxximation
# reminder, the gradient of a image is performed in two dimension
def grad(x):
    grad = np.array(np.gradient(x,edge_order=2))
    return grad
# norm of the gradient
def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


# Function to perform redistancing according to Fast Marching Methods
# $\phi = sign(\phi)(1-||\nabla\phi||)$

# In[68]:


# function to perform redistancing
# uses the skfmm library to perform the redistancing using fast marching method
def redistancing(phi):
    sd = skfmm.distance(phi, dx = 1)
    return sd


# Compute the c level set or intersection of a curve phi.

# In[69]:



# compute the intersection of the level set and the curve phi
# eps is to prevent numerical errors
def intersect(phi,c,eps):
    if phi <= c+eps and phi >=c-eps:
        return 1
    else:
        return 0  


# Use compute F according to $F = \frac{1}{1+||\nabla I||}$

# In[70]:


# stopping function for the level set. 
def stopping_fun(x):
    return 1. / (1. + norm(grad(x))**2)


# Functions to concatenate images

# In[71]:


# concatenate the frames either horizontally or vertically
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


# Get the image and do some basic pre processing

# In[72]:


# get the image in grayscale
img = cv.imread(path_source)
img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# apply gaussian filter to img
img_gauss = ndimage.gaussian_filter(img, sigma=1)
# center on the mean
img_mean = img_gauss - np.mean(img_gauss)


# Set the x and y axis to the size of the image.
# x and y are a a np.mesh, which is similar to a cartesian grid

# In[73]:


# limits of the mesh, x start, x end, y start, y end 
limits = [-int(img.shape[1]/2),int(img.shape[1]/2),-int(img.shape[0]/2),int(img.shape[0]/2)]
# extend to plot
extent = [-int(img.shape[1]/2),int(img.shape[1]/2),-int(img.shape[0]/2),int(img.shape[0]/2)]
# x = np.asarray([i for i in range(img.shape[0])] )
# y = np.asarray([i for i in range(img.shape[1])] )
x = np.linspace(-int(img.shape[1]/2),int(img.shape[1]/2)-1, img.shape[1])
y = np.linspace(-int(img.shape[0]/2),int(img.shape[0]/2)-1, img.shape[0])
x,y = np.meshgrid(x,y)


# Set an initial shape of the curve phi

# In[74]:


# initial shape
# cone in this case. It's level set is a circle
phi = np.sqrt((x**2+y**2))-1
# phi = phi-np.mean(z)


# Set the initial shape as the distance from its zero level set

# In[75]:


# phi surface function as distance from level set 0 the the image
phi = redistancing(phi)


# Set the velocity F as $F = \frac{1}{1+||\nabla I||}$ of the image I

# In[76]:


# F velocity field according to gradient of mean image
F = np.ones(img.shape)
F = stopping_fun(img_mean) * v


# Frame counters to make the final animation

# In[77]:


#frame count
frame_draw = [i*n_iter/frames for i in range(frames+1)]
fr_count = 0


# Iteratively update $\phi$ as $\phi(x,y,t+\Delta t) = \phi(x,y,t) - \Delta t F||\nabla\phi||$

# In[78]:


for i in range(n_iter):
    # if redistancing is allowed
    if i%redistancing_iter == 0 and allow_redistancing:
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

    # if the frame count is in the list of frames to be drawn
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
        print("Frame number {}".format(fr_count))
        fr_count+=1


# Concatenates the images to make a final gif

# In[82]:


# concatenates the frames and make a gif        
im_frame = []
for c in range(0,frames,step):
    im1 = Image.open(path.format(c))
    # im1 = im1.resize((1120, 840),)
    im2 = Image.open(path_contour.format(c))
    # im2 = im2.resize((1120, 840),)
    if vertical_concat:
        im3 = get_concat_v(im2, im1)
    else:
        im3 = get_concat_h(im2, im1)
    im_frame.append (im3)

im_frame[0].save(path_gif, save_all=True, append_images=im_frame[1:], duration=100, loop=0)


# Show the final animation. Change the markdown to the correct path of the gif 

# <img src='../images/brain tumor/out.gif' width="1000" align="center">
