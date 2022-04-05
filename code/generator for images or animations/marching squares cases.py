# create the image of cases for marching squares

import numpy as np
import matplotlib.pyplot as plt
import itertools
# cases for the lines to draw
def line(x,y,c):
    draw = True
    draw2 = False
    y2 = np.empty(len(x))
    if c[0] == False and c[1] == False and c[2] == False and c[3] == False:
        y = np.empty(len(x))
        draw = False
    if c[0] == False and c[1] == False and c[2] == False and c[3] == True:
        # draw a line 45 degree
        y = -x+1.5
    if c[0] == False and c[1] == False and c[2] == True and c[3] == False:
        # draw a line 45 degree
        y = x-0.5
    if c[0] == False and c[1] == False and c[2] == True and c[3] == True:
        # draw a line 45 degree
        x = np.asarray([0.5 for i in x])
    if c[0] == False and c[1] == True and c[2] == False and c[3] == False:
        # draw a line 45 degree
        y = x+0.5
    if c[0] == False and c[1] == True and c[2] == False and c[3] == True:
        # draw a line 45 degree
        y = np.asarray([0.5 for i in x])
    if c[0] == False and c[1] == True and c[2] == True and c[3] == False:
        # draw a line 45 degree
        y = -x+0.50
        y2 = -x+1.50
        draw2 = True
    if c[0] == False and c[1] == True and c[2] == True and c[3] == True:
        # draw a line 45 degree
        y = -x+0.25
    
    if c[0] == True and c[1] == False and c[2] == False and c[3] == False:
        # draw a line 45 degree
        y = -x+0.5
    if c[0] == True and c[1] == False and c[2] == False and c[3] == True:
        # draw a line 45 degree
        y = x-0.50
        y2 = x+0.50
        draw2 = True
    if c[0] == True and c[1] == False and c[2] == True and c[3] == False:
        # draw a line 45 degree
        y = np.asarray([0.5 for i in x])
    if c[0] == True and c[1] == False and c[2] == True and c[3] == True:
        # draw a line 45 degree
        y = x+0.5
    if c[0] == True and c[1] == True and c[2] == False and c[3] == False:
        # draw a line 45 degree
        x = np.asarray([0.5 for i in y])
    if c[0] == True and c[1] == True and c[2] == False and c[3] == True:
        # draw a line 45 degree
        y = x-0.5
    if c[0] == True and c[1] == True and c[2] == True and c[3] == False:
        # draw a line 45 degree
        y = -x+1.5
    if c[0] == True and c[1] == True and c[2] == True and c[3] == True:
        y = np.empty(len(x))
        draw = False
    return x,y,draw,y2,draw2

# meshgrid for the cases
x = np.asarray([0.0,0.0,1.0,1.0])
y = np.asarray([0.0,1.0,0.0,1.0])

# list of possible combinations of 4 binary values of edges
lst = list(itertools.product([False, True], repeat=4))
c = [list(i) for i in lst]

# plot to show all the cases
fig, axs = plt.subplots(4, 4, figsize=(9, 3), sharey=True)
fig.suptitle('Cases')

xi = 0
yj = 0

for i in range(len(c)):
    ax = axs[xi,yj]
    # set the colors
    color = ['blue' if j else 'black' for j in c[i]]
    # define a np array of equally spaced numbers from 0 to 1
    x_line = np.linspace(-0.5,2,100)
    y_line = np.linspace(-0.5,2,100)
    ax.scatter(x,y,c=color)
    ax.set_xlim(-0.25,1.25)
    ax.set_ylim(-0.25,1.25)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_title(str('case {}'.format(i)))
    # get the cases and plot
    x_line,y_line,draw,y2_line,draw2 = line(x_line,y_line,c[i])  
    if draw : 
        ax.plot(x_line,y_line,color='red')
    if draw2 :
        ax.plot(x_line,y2_line,color='red')
    # ax.axis('off')
    # plt.ion()
    # ax.pause(0.1)
    # ax.show()
    
    # change the position in the plot
    xi = (xi+1)%4
    if xi==0:
        yj = (yj+1)%4
plt.show()
