# draw the circles with arrows for the ppt

import matplotlib.pyplot as plt
from math import sin
from math import cos
from math import pi
from PIL import Image
# number of frames
frames  = 100
# initial size of the circle
initial_size = 1
# step size for the circle
delta_radius = 0.025
#path to save the frames
path ='images/two circles/plotcircles {}.png'
#path where to save the gif
path_gif='images/two circles/plotcircles.gif'
for i in range(frames):
    # circles to plot
    size = initial_size + delta_radius*i
    circle1 = plt.Circle((0., 0.), size, color='blue', fill=False)
    circle2 = plt.Circle((3, 3),  size, color='blue',fill=False)
    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    ax.axis('equal')
    # ax.axis('off')
    # ax.axis(['-10.', '10.', '-10.', '10.'])
    plt.ylim([-1,5])
    plt.xlim([-1,5])


    ax.add_patch(circle1)

    # arrows
    plt.arrow(size *cos(pi/4), size*sin(pi/4), .5, .5, head_width=.15, color='r', length_includes_head=True)
    plt.arrow(size*cos(pi/2), size*sin(pi/2), .0, .5, head_width=.15, color='r', length_includes_head=True)
    plt.arrow(size*cos(0), size*sin(0), .5, .0, head_width=.15, color='r', length_includes_head=True)
    plt.text(size*sin(pi/4)+0.5, size*cos(pi/4)+.5, '$F$', fontsize=20)



    ax.add_patch(circle2)

    # #arrows
    plt.arrow(size*sin(pi*5/4)+3, size*cos(pi*5/4)+3, -.5, -.5, head_width=.15, color='r', length_includes_head=True)
    plt.arrow(size*cos(pi*3/2)+3, size*sin(pi*3/2)+3, .0, -.5, head_width=.15, color='r', length_includes_head=True)
    plt.arrow(size*cos(pi)+3, size*sin(pi)+3, -.5, .0, head_width=.15, color='r', length_includes_head=True)

    # fig.show()
    fig.savefig(path.format(i))
    plt.close()
    
im_frame = []
for c in range(frames):
    im1 = Image.open(path.format(c))    
    im_frame.append (im1)
im_frame[0].save(path_gif, save_all=True, append_images=im_frame[1:], duration=1, loop=0)