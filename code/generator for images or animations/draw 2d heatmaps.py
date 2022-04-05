# draw the heatmaps for the animations 

import enum
from math import sqrt
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from numpy import cos, sin
from numpy import exp
import time


# limits of the mesh, x start, x end, y start, y end 
limits = [-5,5,-5,5]
# extend of the heatmap to plot
extent = [-5,5,-5,5]
# resolution of the meshgrid in both sizes
resolution = 1000
# vertical starting point
v_start = 0
# vertical resolution
v_res = 100
# vertical max value
v_max = 10
# path of the images
path = "./images/cones/heatmap {}.png"


# function to draw the heatmaps for the animations
def sphere(z,x,y):
    for i,x_i in enumerate(x):
        for j,y_j in enumerate(y):
            z[i,j] = x[i]**2+y[j]**2

def sincos(z,x,y):
    for i,x_i in enumerate(x):
        for j,y_j in enumerate(y):
            z[i,j] = cos(x[i])+sin(y[j])+2

def e(z,x,y):
    for i,x_i in enumerate(x):
        for j,y_j in enumerate(y):
            z[i,j] = x_i*exp(-x_i**2 - y_j**2)
    return z     
def cones(z,x,y):
    for i,x_i in enumerate(x):
        for j,y_j in enumerate(y):
            z[i,j] = min(sqrt(x[i]**2+y[j]**2),sqrt((x[i]+1)**2+(y[j]+1)**2))

def intersect(z,c):
    if z <= c+0.005 + 0.005*abs(z) and z >= c-0.005 - 0.005*abs(z):
        return 10000
    
    else:
        return 0 


# Generate the x,y meshgrid
x = np.linspace(limits[0], limits[1], resolution)
y = np.linspace(limits[2], limits[3], resolution).T
z = np.zeros(shape=(resolution,resolution))

# apply the function

# sphere(z,x,y)
# sincos(z,x,y)
cones(z,x,y)

cs = np.linspace(v_start, v_max, v_res)
# cs = [i/10 - v_start for i in range(0,v_res)]

for i,c_val in enumerate(cs):
    
    c = np.ones_like(z)* c_val
    intersection = np.zeros_like(c)

    # compute the intersection or "c level set"
    for k,x_i in enumerate(x):
        for j,y_j in enumerate(y):
            intersection[k,j] = intersect(z[k,j],c[k,j])
            
    # apply a mask
    mask = np.ma.masked_where(intersection == 0, intersection)

    # plot the heatmap
    p = plt.imshow(z,extent=extent)
    plt.colorbar(p)
    # plot the mask on top of the heatmap
    p = plt.imshow(mask,extent=extent,interpolation='none',cmap='cool')
    # save the image
    plt.savefig(path.format(i))
    # show the image
    plt.show()
