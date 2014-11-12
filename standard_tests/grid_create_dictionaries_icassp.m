

id_m = [1,2,3];
id_f = [4,7,11];

id = [id_m,id_f];

use_spect = 1;
use_scatt2 = 0;


if use_scatt2
    
    representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/scatt2_fs16_NFFT2048/';
    
if ~exist('stds1','var')
X1 = [];
X2 = [];
for i=1:length(id)
    load(sprintf('%ss%d',representation,id(i)));
    X1 = [X1 data.X1];
    X2 = [X2 data.X2];
end

stds1 = std(X1,0,2);
stds2 = std(X2,0,2);
end

for i=1:length(id)

  
    
    load(sprintf('%ss%d',representation,id(i)));
    
    Npad = 2^15;
    
    options.renorm=1;
    if options.renorm
        %renormalize data: whiten each frequency component.
        eps=2e-3;
        data.X1 = renorm_spect_data(data.X1, stds1, eps);
        
        eps=1e-3;
        data.X2 = renorm_spect_data(data.X2, stds2, eps);
    end
    
    
    %% train models
    
    model = 'NMF-scatt2';
    
    
    %%%%Plain NMF%%%%%%%
    KK1 = [160];
    LL1 = [0.1];
    param1.K = KK1;
    param1.posAlpha = 1;
    param1.posD = 1;
    param1.pos = 1;
    param1.lambda = LL1;
    param1.iter = 4000;
    param1.numThreads=16;
    param1.batchsize=512;
    
    Dnmf1 = mexTrainDL(abs(data.X1),param1);
    
    KK2 = [768];
    LL2 = [0.1];
    param2.K = KK2;
    param2.posAlpha = 1;
    param2.posD = 1;
    param2.pos = 1;
    param2.lambda = LL2;
    param2.iter = 4000;
    param2.numThreads=16;
    param2.batchsize=512;
    
    Dnmf2 = mexTrainDL(abs(data.X2),param2);
    
    save(['/misc/vlgscratch3/LecunGroup/bruna/grid_data/icassp_grid_exp/dictionaries_NMF_scatt2_s' num2str(id(i)) '.mat'],'Dnmf1','Dnmf2','param1','param2')
    
    disp(i)
    
end

end


if use_spect
    
 representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_fs16_NFFT1024_hop512/';
    
for i=1:length(id)
    
    load(sprintf('%ss%d',representation,id(i)));

    
    param.K = 200;
    param.posAlpha = 1;
    param.posD = 1;
    param.pos = 1;
    param.lambda = 0.1;
    param.lambda2 = 0;
    param.iter = 4000;
    
    
    Dnmf = mexTrainDL(abs(data.X), param);
    
    save(['/misc/vlgscratch3/LecunGroup/bruna/grid_data/icassp_grid_exp/dictionaries_NMF_spect_s' num2str(id(i)) '.mat'],'Dnmf','param')
    
    disp(i)
    
    
end
    
    
end

