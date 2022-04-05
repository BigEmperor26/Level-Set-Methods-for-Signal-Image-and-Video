# combines two images horizontally or vertically and generates a gif
from PIL import Image
# path of first image
path_heatmap = './images/water coin stop/image contour{}.png'
# path of second images
path_surface = './images/water coin stop/image 3D{}.png'
# path to save the final output gif
path_gif = './images/water coin stop/out.gif'
# number of frames for the animation
num_frames = 100
# horizontal or vertical concatenation
horizontal = True
vertical = False

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

cs = [i for i in range(0,num_frames)]
im_frame = []
for c in cs:
    im1 = Image.open(path_heatmap.format(c))
    # im1 = im1.resize((1120, 840),)
    im2 = Image.open(path_surface.format(c))
    # im2 = im2.resize((1120, 840),)
    if horizontal:
        im3  = get_concat_h(im2, im1)
    if vertical:
        im3  = get_concat_v(im2, im1)
    im_frame.append (im3)
# save the image
im_frame[0].save(path_gif, save_all=True, append_images=im_frame[1:], duration=100, loop=0)