clear; close all; clc;

% Create phantom image:
img_size = 256;
P = phantom('Modified Shepp-Logan',img_size);

% Create sinogram:
dtheta = 3;
theta = (0:dtheta:180);
theta = theta(1:end-1);
y = radon(P,theta);

% % Add Gaussian noise:
noise = 2*randn(size(y));
y = y + noise;

% Reconstruction with filtered backprojection:
x_filt_backp = iradon(y,theta,'linear','ram-lak',1,img_size);

x0 = zeros(img_size);
iters  = 2000;

% QCBP Fourier reconstruction:
xi0 = zeros(size(y));
tau   = 1e-7; % needs to be much smaller than sigma
sigma = 1e+3;
eta = norm(noise,'fro');
[x_cs,xi] = QCBP_fourier(x0,xi0,tau,sigma,eta,y,iters, true);


plot_results(P,x_filt_backp,x_cs)





















