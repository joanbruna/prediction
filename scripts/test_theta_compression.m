if ~exist('X1','var')
    % use single speaker for training
    load ../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s31.mat
    X1 = Xc;
    clear Xc;
    
    epsilon = 1;
    X1 = softNormalize(X1,epsilon);
    
    load ../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s14.mat
    X2 = Xc;

    X2 = softNormalize(X2,epsilon);
    
    
    X = [X1 X2];
    clear X1 X2
    
end



if ~exist('D','var')
    load dict_2_speakers 
end


M = size(X,2);

batchsize = 200;

niters=round(M/batchsize);

%initial dictionary
II=1:batchsize:M;


options.lambda = 0.1;
options.mu = 0.5;

ptheta = struct;
ptheta.sigma = 1;
ptheta.hn = 11;
ptheta.lambda = 0.1;
ptheta.lambdar = 0.00001;

K = size(D,2);
Theta = zeros(K,M);

for i=1:(length(II)-1)
    
    init_batch = II(i);
    I0 = init_batch:(init_batch+batchsize-1);
    data=X(:,I0);
    
    
    [A,theta,SA] = nmf_optflow_smooth(data,D,options,ptheta);
    
    Theta(:,I0) = theta; 
    
%     figure(3)
%     subplot(311)
%     dbimagesc(data+0.001);
%     subplot(312)
%     imagesc(A)
%     subplot(313)
%     imagesc(SA)
%     drawnow

    %save theta_temp Theta
    
    sprintf('Iteration %1.0f of %1.0f\n',i,length(II))
    
end



%%


options.K=50;
options.epochs=2;
options.nmf = 0;
options.alpha_iters=80;
options.batchsize=256;
options.sort_dict = 1;
options.plot_dict = 0;
options.lambda = 0.1;
options.mu = 0.5;

options.init_nmf = 1;
options.use_flow = 1;


[Dslow_,Dnmf_] = train_nmf_optflow(Theta, options);
