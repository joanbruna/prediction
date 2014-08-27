

close all
clear all

%experiments doing a warping by cascading wavelet modulations. 

N=32;
options.J=3;

filters=prepare_filters_2d(N, options);
dualfilters{1}=prepare_dualfilters_2d(filters{1});

[tr]=retrieve_mnist_data(100,1);
f1=12;
f2=5;
ref = tr{3}{f1}';
target = tr{3}{f2}';

%[Uref] = wmod_fprop(ref, filters);
%Utarg = wmod_fprop(target, filters);

second=0;
maxiters=900;
lambda = 5e-1;

%path 
J=size(filters{1}.psi,2);
L=size(filters{1}.psi{1},2);

ftarget = fft2(target);
for j=1:J
    for l=1:L
        M{j}{l}=ifft2(ftarget.*filters{1}.psi{j}{l});
        U{j}{l}=abs(M{j}{l});
        if second
        ftmp = fft2(U{j}{l});
        for jj=j+1:J
            for ll=1:L
                starget{j}{l}{jj}{ll} = abs(ifft2(ftmp.*filters{1}.psi{jj}{ll}));
            end
        end
        starget{j}{l}{J+1} = ifft2(ftmp.*filters{1}.phi);
        else
        starget{j}{l} = U{j}{l};
        end
    end
end
starget{J+1} = ifft2(ftarget .* filters{1}.phi);

npool=0;
tempo{1} = ref;
nnorm = norm(ref(:)-target(:));
rho = 0.04;
i=2;
apprerr(1)=1;
while norm(tempo{i-1}(:)-target(:)) > rho * nnorm && i < maxiters
%for i=2:steps
gout = scatt_grad_step(tempo{i-1}, fft2(tempo{i-1}), starget, filters, dualfilters, second);

%[tx, ty]=gradient(tempo{i-1});
%gg(1,:,:) = -lambda *gout.*tx;
%gg(2,:,:) = -lambda *gout.*ty;
%tempo{i} = warp2d(tempo{i-1},gg);
%tempo{i} = tempo{i-1} + tx.*squeeze(gg(1,:,:)) + ty.*squeeze(gg(2,:,:));

tempo{i} = tempo{i-1} - lambda * gout;
apprerr(i) = norm(tempo{i}(:)-target(:))/nnorm;
    fprintf('progress %f \n', norm(tempo{i}(:)- target(:))/nnorm)
i=i+1;
end

steps=i-1;

%clear starget;
%second=0;
%%maxiters=500;
%lambda = 6e-1;
%
%%pasth 
%J=size(filters{1}.psi,2);
%L=size(filters{1}.psi{1},2);
%
%ftarget = fft2(target);
%for j=1:J
%    for l=1:L
%        M{j}{l}=ifft2(ftarget.*filters{1}.psi{j}{l});
%        U{j}{l}=abs(M{j}{l});
%        if second
%        ftmp = fft2(U{j}{l});
%        for jj=j+1:J
%            for ll=1:L
%                starget{j}{l}{jj}{ll} = abs(ifft2(ftmp.*filters{1}.psi{jj}{ll}));
%            end
%        end
%        starget{j}{l}{J+1} = ifft2(ftmp.*filters{1}.phi);
%        else
%        starget{j}{l} = U{j}{l};
%        end
%    end
%end
%starget{J+1} = ifft2(ftarget .* filters{1}.phi);
%
%npool=0;
%tempo1{1} = ref;
%nnorm = norm(ref(:)-target(:));
%rho = 0.1;
%%i=2;
%%while norm(tempo1{i-1}(:)-target(:)) > rho * nnorm && i < maxiters
%for i=2:steps
%gout = scatt_grad_step(tempo1{i-1}, fft2(tempo1{i-1}), starget, filters, dualfilters, second);
%tempo1{i} = tempo1{i-1} - lambda * gout;
%    fprintf('progress %f \n', norm(tempo1{i}(:)- target(:))/nnorm)
%%i=i+1;
%end
%



%steps=i-1;

%create movie 
clear mov;
figure, set(gcf, 'Color', 'white')
imagesc([-ref -ref]); axis tight
set(gca, 'nextplot', 'replacechildren', 'Visible', 'off');
tss=4;
nFrames=floor(steps/tss)/2;
mov(1:nFrames) = struct('cdata',[],'colormap',[]);

for k=1:nFrames
    %imagesc([-tempo{k} -(k/nFrames)*target - ((nFrames-k)/nFrames)*ref]);colormap gray;
    imagesc([ -(1-apprerr(tss*k))*target - apprerr(tss*k)*ref -tempo{tss*k} ]);colormap gray;
    mov(k) = getframe(gca);
end

movie2avi(mov, 'mnistmov.avi', 'compression', 'None', 'fps', 10);


%check dual filters
x=randn(N);
xrec=0*x;
fx=fft2(x);
for j=1:J
    for l=1:L
        tmp = ifft2(fx.*filters{1}.psi{j}{l}.*dualfilters{1}.psi{j}{l});
        xrec = xrec + tmp;
     end
end
tmp = ifft2(fx.*filters{1}.phi.*dualfilters{1}.phi);
xrec = real(xrec + tmp);



