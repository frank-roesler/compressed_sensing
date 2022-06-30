function plot_results(P,x_filt_backp,xf_fista)
figure('Position',[300,200,1500,400])
subplot(1,3,1)
imagesc(P)
caxis([0,1])
colorbar
title('Original')
axis off

subplot(1,3,2)
imagesc(x_filt_backp)
caxis([0,1])
colorbar
title('Filtered backprojection')
axis off

subplot(1,3,3)
imagesc(real(fft2(xf_fista)))
caxis([0,1])
colorbar
title('Compressed sensing (Fourier FISTA)')
axis off
colormap gray
end