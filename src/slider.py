import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button

#Setup figure and data
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
delta = 0.5

t = np.linspace(0.0, 100.0, 256)

x = np.linspace(-4.0, 4.001, 512)
y = np.linspace(-4.0, 4.001, 512)
X, Y = np.meshgrid(x, y)

#Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
#Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
#x, y = np.random.default_rng().multivariate_normal(mean, cov, 5000).T
Z1 = np.random.default_rng().multivariate_normal([0,1], [[0.5,0.3], [0.3,1.]],(len(x), len(y)))
XZslice = np.zeros((256,512,512))
for i in range(t.shape[0]):
    XZslice[i,:,:] = (Z1[:,:,0] - Z1[:,:,1]) * t[i]/10.
cmap = plt.cm.rainbow
im = ax.pcolormesh(XZslice[128,:,:], cmap=cmap)
fig.colorbar(im)
axcolor = 'lightgoldenrodyellow'
axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
stime = Slider(axtime, 'Time', 0.0, 100.0, valinit=50.0)

#Routines to reset and update sliding bar
def reset(event):
    stime.reset()

def update(val):
    time = int(stime.val/100.* 256)
    im.set_array(XZslice[time,:,:].ravel())
    fig.canvas.draw()

#Bind sliding bar and reset button  
stime.on_changed(update)
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
button.on_clicked(reset)

plt.show()