
# generates animation from frames to a gif
from PIL import Image

# path of source frames
path_frame = "images/3d test/image{}.png"
# path of output frames
path_gif = "images/3d test/out.gif"
# number of frames
frame = 100
#step how many frames to skip
step= 1
cs = [i for i in range(0,frame,step)]
im_frame = []
for c in cs:
    im1 = Image.open(path_frame.format(c))    
    im_frame.append (im1)

im_frame[0].save(path_gif, save_all=True, append_images=im_frame[1:], duration=1, loop=0)