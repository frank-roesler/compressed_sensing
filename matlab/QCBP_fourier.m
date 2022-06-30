function [x,xi] = QCBP_fourier(x0,xi0,tau,sigma,eta,y,iters, plotting)
% Implementation of a QCBP minimization algorithm for compressed sensing with fft2
    dtheta = 180/size(y,2);
    theta = 0:dtheta:180;
    if theta(end)==180
        theta = theta(1:end-1);
    end
    img_size = size(x0,1);
    x = x0;
    x_test_old = x0;
    xi = xi0;
    figure('Position',[300,200,1000,400])
    for n=1:iters
        xold = x;
        Re = iradon(real(xi),theta,'linear','none',1,img_size);
        Im = iradon(imag(xi),theta,'linear','none',1,img_size);
        x = soft_threshold(x - tau*ifft2(Re + 1i*Im), tau);
        zeta = xi + sigma*radon(fft2(2*x-xold),theta) - sigma*y;
        if norm(zeta,'fro') <= sigma*eta
            xi = zeros(size(xi0));
        else
            xi = (1-sigma*eta/norm(zeta,'fro'))*zeta;
        end
        
        % Plot results:
        if mod(n,1000)==0 && plotting
            subplot(1,2,1)
            imagesc(real(fft2(x)))
            title('Real part')
            caxis([0,1])
            colorbar
            axis off
            subplot(1,2,2)
            imagesc(imag(fft2(x)))
            title('Imaginary part')
            caxis([0,1])
            colorbar
            axis off
            colormap('gray')
            drawnow
            n
            error = max(max(abs(x_test_old-x)))
            x_test_old = x;
        end
    end
end

function st = soft_threshold(x,tau)
    st = zeros(size(x));
    idx = abs(x)>=tau;
    st(idx) = sign(x(idx)).*(abs(x(idx))-tau);
end