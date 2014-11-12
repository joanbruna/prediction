
tol = 1e-4;
n_iter_max = 500;
beta = 1;

rho = 1;

l_win = 1024;
overlap = l_win/2;
Fs = 16000;

lambda_ast = 0;
lambda = 0;


%% load data

representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_fs16_NFFT1024_hop512/';

id_1 = 2;
id_2 = 11;

% another man!
%id_2 = 14;


load(sprintf('%ss%d',representation,id_1));
data1 = data;
clear data

idx = randperm(size(data1.X,2));
data1.X = data1.X(:,idx(1:10000));

load(sprintf('%ss%d',representation,id_2));
data2 = data;
clear data

idx = randperm(size(data2.X,2));
data2.X = data2.X(:,idx(1:10000));



%% train models

model = 'NMF-KL';

KK = [10,30,50];
LL = [0,0.1];

for ii = 1:length(KK)

for jj = 1:length(LL)

    param.K = KK(ii);
    param.posAlpha = 1;
    param.posD = 1;
    param.pos = 1;
    param.lambda = LL(jj);
    param.lambda2 = 0;
    param.iter = 500;
    
    K =param.K;
    
    [F,N] = size(data1.X);
    
    W_ini = abs(randn(F,K)) + 1;
    H_ini = abs(randn(K,N)) + 1;
    E_ini = zeros(F,N);
    
    %D1 = mexTrainDL(abs(data1.X), param);
    D1 = rnmf(abs(data1.X), beta, n_iter_max, tol, W_ini, H_ini, E_ini, param.lambda2,param.lambda,1);
    
    [F,N] = size(data1.X);
    
    W_ini = abs(randn(F,K)) + 1;
    H_ini = abs(randn(K,N)) + 1;
    E_ini = zeros(F,N);
    
    
    %D2 = mexTrainDL(abs(data2.X), param);
    D2 = rnmf(abs(data2.X), beta, n_iter_max, tol, W_ini, H_ini, E_ini, param.lambda2,param.lambda,1);
    
    model_name = sprintf('%s-K%d-lambda%d-lambda2%d',model,param.K,round(10*param.lambda),round(10*param.lambda2));
    
    
    %% test models

    % saving setting

    save_folder = sprintf('../../public_html/speech/%s-s%d-s%d-%s/',model_name,id_1,id_2,date());

    try
        unix(sprintf('mkdir %s',save_folder));
        unix(sprintf('chmod 777 %s ',save_folder));
    catch
    end

    %%


    NFFT = data1.NFFT;
    fs = data1.fs;
    hop = data1.hop;
    N_test = 200;
    SDR = 0;
    SIR = 0;
    
    for i = 1:N_test

        [x1, Fs] = audioread(sprintf('%s%s',data1.folder,data1.d(data1.testing_idx(i) ).name) );
        x1 = resample(x1,fs,Fs);
        x1 = x1(:)'; T1 = length(x1);


        [x2, Fs] = audioread(sprintf('%s%s',data2.folder,data2.d(data2.testing_idx(i) ).name) );
        x2 = resample(x2,fs,Fs);
        x2 = x2(:)'; T2 = length(x2);

        T = min(T1,T2);

        x1 = x1(1:T);
        x2 = x2(1:T);

        %x1 = x1/norm(x1);
        %x2 = x2/norm(x2);

        mix = (x1+x2);

        X = compute_spectrum(mix,NFFT,hop);

        [F,N] = size(X);
        
        % compute decomposition
        H_ini = abs(randn(2*K,N)) + 1;
        E_ini = zeros(F,N);

        %[~, H, E, obj, fit, V_ap] = rnmf(V, beta, n_iter_max, tol, [W1,W2], H_ini, E_ini, lambda_ast,lambda,0);
        [~,H] = nmf_admm(abs(X), [D1,D2], H_ini, beta, rho,1:2*K);
        %H =  full(mexLasso(abs(X),[D1,D2],param));

        W1H1 = D1*H(1:size(D1,2),:);
        W2H2 = D2*H(size(D1,2)+1:end,:);

        eps = 1e-6;
        V_ap = W1H1 +W2H2 + eps;

        % wiener filter

        SPEECH1 = ((W1H1.^2)./(V_ap.^2)).*X;
        SPEECH2 = ((W2H2.^2)./(V_ap.^2)).*X;
        speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
        speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);

        Parms =  BSS_EVAL(x1', x2', speech1', speech2', mix');
        
        NSDR = SDR+mean(Parms.NSDR)/N_test;
        SIR = SIR+mean(Parms.SIR)/N_test;
        
        Parms
        output{i} = Parms;

        file1 = sprintf('%s%dspeech-1.wav',save_folder,i);
        audiowrite(file1,speech1,fs);
        unix(sprintf('chmod 777 %s',file1));

        file2 = sprintf('%s%dspeech-2.wav',save_folder,i);
        audiowrite(file2,speech2,fs);
        unix(sprintf('chmod 777 %s',file2));

        filemix = sprintf('%s%dmix.wav',save_folder,i);
        audiowrite(filemix,mix,fs);
        unix(sprintf('chmod 777 %s',filemix));

    end
    save_file = sprintf('%sresults.mat',save_folder,'s');
    save(save_file,'output','D1','D2','param','NSDR','SIR')
    unix(sprintf('chmod 777 %s ',save_file));
    AA{ii,jj}.res = output;
    clear output
end
end

