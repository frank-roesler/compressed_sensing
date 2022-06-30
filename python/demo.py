from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from cs_algorithms import QCBP_fourier
from numpy.fft import fft2


original = shepp_logan_phantom()
original = rescale(original, scale=0.6, mode='reflect', channel_axis=None)
img_size = original.shape[0]

dtheta = 2
theta = np.arange(0,180,dtheta)
sinogram = radon(original,theta)

noise = 2*np.random.randn(*sinogram.shape)
sinogram += noise

x_filt_backp = iradon(sinogram, theta=theta, output_size=img_size, filter_name='ramp')

x0 = np.zeros((img_size,img_size), dtype=np.complex64)
iters  = 1000
xi0 = np.zeros(sinogram.shape, dtype=np.complex64)
tau   = 1e-3
sigma = 2e+0
eta = norm(noise)
[x_cs,xi] = QCBP_fourier(x0,xi0,tau,sigma,eta,sinogram,iters)


# Plot results
fig, (ax1,ax2) = plt.subplots(1,2,constrained_layout=True)
p1=ax1.imshow(np.real(fft2(x_cs)), cmap=plt.cm.Greys_r)
p2=ax2.imshow(x_filt_backp, cmap=plt.cm.Greys_r)
p1.set_clim(0, 1)
p2.set_clim(0, 1)
ax1.set_title('Compressed Sensing')
ax2.set_title('Filtered Backprojection')

plt.show()








