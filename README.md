# Compressed Sensing for the Radon transform

Matlab implementation of an algorithm to solve the [Quadratically Constrained Basis Pursuit (QCBP)](https://en.wikipedia.org/wiki/Basis_pursuit) problem. The algorithm solves the problem 
$$\min\_{x\in\mathbb{C}^{N\times N}} \Vert\Phi^* x\Vert\_{l^1} \quad \text{subject to} \quad \Vert Ax-y\Vert\_{l^2}<\\|e\\|\_{l^2},$$
where $x$ 
denotes a 
$N\times N$ 
matrix (representing an image), 
$A$ 
denotes the Radon transform, 
$\Phi$ 
the Fourier transform, 
$y$
the measured data (sometimes called a sinogram) and 
$e$
denotes Gaussian noise.  
In words, the above minimization process chooses among all solutions of $\Vert Ax-y\Vert\_{l^2}<\\|e\\|\_{l^2}$ the one whose Fourier transform is approximately sparse.

### Contents:
* `QCBP_fourier.m` Minimization algorithm based on the gradient descent method,
* `demo.m` Demo script that compares the above method to classical filtered backprojection.

### Dependency:
* Matlab's Image processing toolbox

Any comments or queries are welcome at https://frank-roesler.github.io/contact/
