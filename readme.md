# CODE for Signal Image and Video
## by michele yin

In the folder **code** there is the code I used for this project

In the subfolder **code/generator for images or animations** are some python scripts that generated animations used in the presentations

The main scripts used are called **level set { ... } .py**

- ```Level set without redistancing.py``` propagates a level set without doing redistancing and saves an animation

- ```Level set with redistancing.py``` propagates a level set doing redistancing and saves an animation

- ```Level set with gradient.py``` propagates a level set doing redistancing from an image source. The propagations halts at the edges of the image. Saves an animation

- ```Level set notebook.ipynb``` propagates a level set doing redistancing from an image source. The propagations halts at the edges of the image. Saves an animation. Because is it a jupyter notebook it may be more readable and may be used for an eventual lab

In the folder **images** there are the images that were generated by various scripts

- ```Level set with gradient video.py``` I tried to do in a video, but failed miserably
## Requirements

- Python 3.8
- numpy 1.21.5
- opencv-python-4.5.5.64
- scipy-1.8.0
- skfmm 2022.3.26
- Pillow 9.0.0
- matplotlib 3.5.1

To install these if you don't have run
```
pip install numpy
pip install python-opencv
pip install cipy
pip install scikit-fmm
pip install Pillow
pip install matplotlib
```
## What are level set ?

A level set of a function <img src="https://render.githubusercontent.com/render/math?math=\phi"> is a 

<img src="https://render.githubusercontent.com/render/math?math=\{(x,y): \phi (x,y) = c\}">


Usually we are insterested in the *zero level set* <img src="https://render.githubusercontent.com/render/math?math=\{(x,y): \phi (x,y) = 0\}">

Here there is a animation of the zero level set of a dynamic curve

<img src="images\circle expanding\out.gif" />

They can be used to model dynamic curves with changing topology

<img src="images\two circles\out.gif" />

The curve <img src="https://render.githubusercontent.com/render/math?math=\phi"> moves according to a velocity <img src="https://render.githubusercontent.com/render/math?math=F">.

Require to solve a **PDE**

<img src="https://render.githubusercontent.com/render/math?math=\frac{ \partial\phi}{\partial t} = F||\nabla \phi||">

Approximate solution by **finite differences**

<img src="https://render.githubusercontent.com/render/math?math=\phi(x,y,t+\Delta t) = \phi(x,y,t) - \Delta t F||\nabla\phi||">

But finite differences have approxximation errors

<img src="images\level set without redistancing\out.gif" />

The curve degrades after a while due to numerical errors propagating

A solution is to reinitialize the curve <img src="https://render.githubusercontent.com/render/math?math=\phi"> every a number of iterations as the signed difference

<img src="https://render.githubusercontent.com/render/math?math=\phi = sign(\phi)(1 - ||\nabla\phi||)">

Computation of this quantity is done through the library <code> skfmm </code>

<img src="images\level set redistancing\out.gif" />

Other implementations do not require redistancing/reinitialization

For image processing we use a velocity <img src="https://render.githubusercontent.com/render/math?math=F"> that is 0 or close to zero at the edges

<img src="https://render.githubusercontent.com/render/math?math=F=\frac{1}{ 1 %+ ||\nabla I||}">

Where <img src="https://render.githubusercontent.com/render/math?math=I"> is the image, and <img src="https://render.githubusercontent.com/render/math?math=\nabla I"> are the edges according to gradient edge detection

This way we the get contours of the image

<img src="images\test images\water_coins.jpg" />

<img src="images\water coins\out.gif" />

<img src="images\lenna\out.gif" />

<img src="images\brain tumor\out.gif" />

# SOURCES

https://en.wikipedia.org/wiki/Marching_square
https://en.wikipedia.org/wiki/Fast_marching_method
https://en.wikipedia.org/wiki/Active_contour_model
https://en.wikipedia.org/wiki/Eikonal_equation
https://en.wikipedia.org/wiki/Level-set_method
https://math.berkeley.edu/~sethian/2006/Explanations/level_set_explain.html
https://math.berkeley.edu/~sethian/2006/Papers/sethian.fastmarching.pdf
https://math.berkeley.edu/~sethian/Papers/sethian.osher.88.pdf
https://math.mit.edu/classes/18.086/2007/levelsetpres.pdf
https://math.mit.edu/classes/18.086/2006/am57.pdf
https://profs.etsmtl.ca/hlombaert/levelset/
https://www.researchgate.net/publication/290437036_A_level_set_method_using_the_signed_distance_function
https://www.sciencedirect.com/science/article/pii/S0021999117307441#br0070
https://agustinus.kristia.de/techblog/2016/11/05/levelset-method/
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.11083.7076&rep=rep1&type=pdf
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5557813



