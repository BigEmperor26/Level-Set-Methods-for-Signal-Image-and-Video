# create a marching squares animation

from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv


size  = 50 # size of the meshgrid
limits = [-5,5,-5,5] # limits of the grid x start, x end, y start, y end
extent = [-6,6,-6,6] # extent of the grid to show x start, x end, y start, y end
vertical_resolution = 50 # vertical resolution of the level sets intersection
v_max = 10 # vertical max value
vertical_start = -10 # from where to start in vertical direction
c_val = 0 # which level set
show_height = False # show height of the level set
precise = True # place the numbers exactly on the grid, if false, they will be centered in each meshgrid square
countour = True # draw contours according to marching blocks
color_mask = True # apply a blue color for squares above the level set
animation = True # animate
save_animation = False # save the animation
path = '../images/marching block/marching frame detail{}.png' # path to save the frames
path_gif = "../images/marching block/marching frame detail.gif" # path to save the gif
path_final = "images/marching block/marching frame final.png" # path to save the final image



def marching_square(mg,c):
    
    x = np.linspace(mg[0],mg[1],2)
    x2 = np.linspace(mg[0],mg[1],2)
    y = np.linspace(mg[2],mg[3],2)
    y2 = np.linspace(mg[2],mg[3],2)
    draw = True
    draw2 = False
    center = [(mg[0]+mg[1])/2,(mg[2]+mg[3])/2]
    bottom = [center[0],center[1]-(mg[3]-mg[2])/2]
    top = [center[0],center[1]+(mg[3]-mg[2])/2]
    left = [center[0]-(mg[1]-mg[0])/2,center[1]]
    right = [center[0]+(mg[1]-mg[0])/2,center[1]]
    # plt.plot(center[0],center[1],'ro')
    # plt.text(center[0],center[1],str(int(center[0]))+","+str(int(center[1])))
    # plt.plot(bottom[0],bottom[1],'ro',color='orange',label='center')
    # plt.plot(top[0],top[1],'ro',color='yellow',label='center')
    # plt.plot(left[0],left[1],'ro',color='yellow',label='center')
    # plt.plot(right[0],right[1],'ro',color='yellow',label='center')
    if c[0] == False and c[1] == False and c[2] == False and c[3] == False:
        # y = np.empty(len(x))
        draw = False
    if c[0] == False and c[1] == False and c[2] == False and c[3] == True:
        x = np.linspace(top[0],right[0],2)
        y = np.linspace(top[1],right[1],2)
    if c[0] == False and c[1] == False and c[2] == True and c[3] == False:
        x = np.linspace(left[0],top[0],2)
        y = np.linspace(left[1],top[1],2)
    if c[0] == False and c[1] == False and c[2] == True and c[3] == True:
        x = np.linspace(left[0],right[0],2)
        y = np.linspace(left[1],right[1],2)
    if c[0] == False and c[1] == True and c[2] == False and c[3] == False:
        x = np.linspace(bottom[0],right[0],2)
        y = np.linspace(bottom[1],right[1],2)
    if c[0] == False and c[1] == True and c[2] == False and c[3] == True:
        x = np.linspace(bottom[0],top[0],2)
        y = np.linspace(bottom[1],top[1],2)
    if c[0] == False and c[1] == True and c[2] == True and c[3] == False:
        x = np.linspace(bottom[0],left[0],2)
        x2 = np.linspace(right[0],top[0],2)
        y = np.linspace(bottom[1],left[1],2)
        y2 = np.linspace(right[0],top[0],2)
        draw2=True
        # 
    if c[0] == False and c[1] == True and c[2] == True and c[3] == True:
        x = np.linspace(bottom[0],left[0],2)
        y = np.linspace(bottom[1],left[1],2)
    if c[0] == True and c[1] == False and c[2] == False and c[3] == False:
        x = np.linspace(bottom[0],left[0],2)
        y = np.linspace(bottom[1],left[1],2)
    if c[0] == True and c[1] == False and c[2] == False and c[3] == True:
        x = np.linspace(bottom[0],right[0],2)
        x2 = np.linspace(left[1],top[1],2)
        x = np.linspace(bottom[0],right[0],2)
        y = np.linspace(left[1],top[1],2)
        draw2 = True
        #
    if c[0] == True and c[1] == False and c[2] == True and c[3] == False:
        x = np.linspace(bottom[0],top[0],2)
        y = np.linspace(bottom[1],top[1],2)
    if c[0] == True and c[1] == False and c[2] == True and c[3] == True:
        x = np.linspace(bottom[0],right[0],2)
        y = np.linspace(bottom[1],right[1],2)
    if c[0] == True and c[1] == True and c[2] == False and c[3] == False:
        x = np.linspace(left[0],right[0],2)
        y = np.linspace(left[1],right[1],2)
    if c[0] == True and c[1] == True and c[2] == False and c[3] == True:
        x = np.linspace(left[0],top[0],2)
        y = np.linspace(left[1],top[1],2)
    if c[0] == True and c[1] == True and c[2] == True and c[3] == False:
        x = np.linspace(right[0],top[0],2)
        y = np.linspace(right[1],top[1],2)
    if c[0] == True and c[1] == True and c[2] == True and c[3] == True:
        # y = np.empty(len(x))
        # draw = False
        draw = False
    return x,y,draw,x2,y2,draw2
    pass

# sphere as a function defined on a meshgrid
sphere = lambda mg: mg[0]** 2 + mg[1] ** 2 - 10

# threshold
def intersect(z,c):
   return True if z >= c else False

# create the matrix
x = np.linspace(limits[0], limits[1], size)
y = np.linspace(limits[2], limits[3], size)
# meshgrid
mg = np.meshgrid(x,y)
# apply the function
z = sphere(mg)

# compute the level sets
cs = [i*v_max/vertical_resolution - vertical_start for i in range(0,vertical_resolution)]

# create a mask for the level set
intersection = np.zeros(shape=(size,size))

# plot the matrix as heatmat
fig, ax = plt.subplots()
# define the colormap
cmap = colors.ListedColormap(['white'])
# draw the function values
ax.matshow(z,extent=extent,cmap=cmap)
ax.axis('off')
if show_height:
    # add label of the vertical value on the heat map
    for i in range(len(x)):
        for j in range(len(y)):
            c = int(z[i, j])
            if precise:
                ax.text( x[i], y[j], str(c), va='center', ha='center')
            else:
                ax.text( ( (limits[1]-limits[0])/size) *(i+1/2) + limits[0], ( (limits[3]-limits[2])/size) *(j+1/2) + limits[2], str(c), va='center', ha='center')
            

# compute the level set
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        intersection[i,j] = intersect(z[i,j],0)

# apply a color mask on top of the function
if color_mask:
    # apply the level set intersection
    c = ['Greys']
    mask = np.ma.masked_where(intersection == False, intersection)
    plt.imshow(mask,extent=limits,interpolation='none',cmap='cool')
frame = 0
# draw the contour
if countour:
    for i in range(0,z.shape[0]-1,1):
        for j in range(0,z.shape[1]-1,1):
            mg = [ x[i], x[i+1],y[j], y[j+1] ]
            c  = [i==1 for i in [intersection[i,j],intersection[i+1,j],intersection[i,j+1],intersection[i+1,j+1]]]
            x_line,y_line,draw,x2_line,y2_line,draw2 = marching_square(mg,c)  
            if draw:
                plt.plot(x_line,y_line,color='red')
            # animate
            if animation:
                points = plt.scatter([x[i],x[i],x[i+1],x[i+1]],[y[j],y[j+1],y[j],y[j+1]],marker='o',color='orange') 
                # save the animation     
                if save_animation:
                    plt.savefig(path.format(frame))
                frame +=1
                points.remove()  
            
            #plt.show()
# save the animation and combinen into a gif
if animation and save_animation:
    cs = [i for i in range(0,frame)]
    im_frame = []
    for c in cs:
        im1 = Image.open(path.format(c))
        
        im_frame.append (im1)

    im_frame[0].save(path_gif, save_all=True, append_images=im_frame[1:], duration=5, loop=0)
    plt.show()
# save final image
plt.savefig(path_final)