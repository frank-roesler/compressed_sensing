from skimage.transform import radon, iradon
import numpy as np
from numpy.fft import fft2, ifft2
from numpy.linalg import norm
import matplotlib.pyplot as plt


def soft_threshold(x,tau):
    """element-wise soft threshold function with parameter tau."""
    st = np.zeros(x.shape)*1j
    idx = np.abs(x)>=tau
    st[idx] = x[idx]/np.abs(x[idx])*(np.abs(x[idx])-tau)
    return st


def QCBP_fourier(x0,xi0,tau,sigma,eta,sinogram,iters, plotting=True):
    """Implementation of minimization algorithm to solve the QCBP
    problem for the Radon transform with Fourier sparsification."""
    dtheta = 180 / sinogram.shape[1]
    theta = np.arange(0,180,dtheta)
    img_size = x0.shape[0]
    x = x0

    # circular mask:
    radius = x0.shape[0]/2
    X = np.arange(x0.shape[0])
    XX,YY = np.meshgrid(X,X)
    dists = np.sqrt((XX-radius+0.5)**2 + (YY-radius+0.5)**2)
    idx = dists>=radius-1

    # main loop:
    x_test_old = x0.copy()
    xi = xi0
    if plotting:
        fig, ax1 = plt.subplots()
    for n in range(iters):
        xold = x.copy()
        backp = iradon(np.real(xi), theta=theta, output_size=img_size, filter_name=None) + \
                1j*iradon(np.imag(xi), theta=theta, output_size=img_size, filter_name=None)
        x = soft_threshold(x - tau * ifft2(backp), tau)

        # Set fft(x) to zero outside the reconstruction circle:
        fx = fft2(x)
        fx[idx] = 0
        x = ifft2(fx)
        fxold = fft2(xold)
        fxold[idx] = 0

        zeta = xi + sigma * (radon(np.real(2*fx - fxold), theta=theta) +
                          1j*radon(np.imag(2*fx - fxold), theta=theta)) - sigma * sinogram
        if norm(zeta) <= sigma*eta:
            xi = np.zeros(xi0.shape, dtype=np.complex64)
        else:
            xi = (1-sigma*eta/norm(zeta))*zeta

        error = np.max(np.abs(x_test_old - x))
        plt_interval = 100
        if plotting and n%plt_interval==0 and n>1:
            ax1.cla()
            p1 = ax1.imshow(np.real(fx), cmap=plt.cm.Greys_r)
            p1.set_clim(0, 1)
            ax1.set_title('Reconstruction')
            plt.show(block=False)
            plt.pause(0.001)

            print('Iteration: ',n)
            print('Change between iterations: ',error)
            print(np.min(np.real(fx)), np.max(np.real(fx)))
            print('-'*100)
            x_test_old = x.copy()
        if error<1e-6*plt_interval:
            break

    return x,xi















