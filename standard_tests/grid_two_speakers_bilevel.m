

%% load data

representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_fs16_NFFT1024_hop512/';

id_1 = 2;
id_2 = 6;

% another man!
%id_2 = 14;


load(sprintf('%ss%d',representation,id_1));
data1 = data;
clear data


load(sprintf('%ss%d',representation,id_2));
data2 = data;
clear data

% epsilon = 1;
param.epsilon = 0.1;
epsilon = param.epsilon;
data1.X = softNormalize(data1.X,epsilon);
data2.X = softNormalize(data2.X,epsilon);

param.renorm=0;
param.save_files = 1;


if param.renorm
%renormalize data: whiten each frequency component.
eps=4e-1;
Xtmp=[abs(data1.X) abs(data2.X)];
stds = std(Xtmp,0,2) + eps;

data1.X = renorm_spect_data(data1.X, stds);
data2.X = renorm_spect_data(data2.X, stds);
end

valid.compute_Parms = 1;

% create validation set

    NFFT = data1.NFFT;
    fs = data1.fs;
    hop = data1.hop;
    N_test = 200;
    SDR = 0;
    SIR = 0;



N_valid = 15;

xx1 = [];
xx2 = [];
mmix = [];

for i = 1:N_valid
    
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
    
    xx1 = [xx1 x1];
    xx2 = [xx2 x2];
    mmix = [mmix mix];
    
end

valid.compute_Parms = 1;
valid.x1 = xx1;
valid.x2 = xx2;
valid.mix = mmix;
valid.S = compute_spectrum(mmix,NFFT,hop);

valid.X1 = compute_spectrum(xx1,NFFT,hop);
valid.X2 = compute_spectrum(xx2,NFFT,hop);


%epsilon = 1;
valid.V = softNormalize(abs(valid.S),epsilon);

nn = size(valid.V,2);


% eliminate data used for validation
data1.X = data1.X(:,nn+1:end);
data2.X = data2.X(:,nn+1:end);

%% train models

model = 'NMF-L2-softnorm';

KK = [50];
LL = [0.1];


for ii = 1:length(KK)

for jj = 1:length(LL)

    param.K = KK(ii);
    param.posAlpha = 1;
    param.posD = 1;
    param.pos = 1;
    param.lambda = LL(jj);
    param.lambda2 = 0.001;
    param.iter = 500;
    

    D1 = mexTrainDL(abs(data1.X), param);

    D2 = mexTrainDL(abs(data2.X), param);
    
 
    % bilevel
%     valid.X1 = data1.X(:,1:1000);
%     valid.X2 = data2.X(:,1:1000);
%     valid.S = valid.X1 +valid.X2;
    

    
    
    
%     valid.overlap = hop;
%     valid.x1 = x1;
%     valid.x2 = x2;
%     valid.mix = mi
    options.valid = valid;
    
    
    options.param0 = param;
    options.beta = 2;
    
    options.step0 = 1;
    options.totaliter = 3000;
    
    %[Wv,Wn,fv,r,Wv_max,Wn_max] = nmf_supervised(Xt1,Xt2,W1,W2,options);
    [Db1,Db2,fv,r] = nmf_supervised_complex(data1.X,data2.X,D1,D2,options);
    

    model_name = sprintf('chapter/%s-K%d-lambda%d-lambda2%d',model,param.K,round(10*param.lambda),round(10*param.lambda2));
    


    %% test models

    % saving setting
    param.save_files = 1;
    if param.save_files
        
        save_folder = sprintf('%s-s%d-s%d-%s/',model_name,id_1,id_2,date());
        %save_folder = sprintf('../../public_html/speech/%s-s%d-s%d-%s/',model_name,id_1,id_2,date());
        
        try
            unix(sprintf('mkdir %s',save_folder));
            unix(sprintf('chmod 777 %s ',save_folder));
        catch
        end
        
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
        
        if param.renorm
            Xn = renorm_spect_data(abs(X), stds);
        end
        %epsilon = 1;
        Xn = softNormalize(abs(X),epsilon);
        
        % compute decomposition
        H =  full(mexLasso(Xn,[Db1,Db2],param));

        W1H1 = Db1*H(1:size(Db1,2),:);
        W2H2 = Db2*H(size(Db1,2)+1:end,:);

        eps_1 = 1e-6;
        V_ap = W1H1.^2 +W2H2.^2 + eps_1;

        % wiener filter

        SPEECH1 = ((W1H1.^2)./V_ap).*X;
        SPEECH2 = ((W2H2.^2)./V_ap).*X;
        speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
        speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);

        Parms =  BSS_EVAL(x1', x2', speech1', speech2', mix');
        
        
        Parms
        output{i} = Parms;
        
        %-----------------------------------------------------------------
        % compute decomposition WITHOUT BILEVEL
        H =  full(mexLasso(Xn,[D1,D2],param));

        W1H1 = D1*H(1:size(D1,2),:);
        W2H2 = D2*H(size(D1,2)+1:end,:);

        eps_1 = 1e-6;
        V_ap = W1H1.^2 +W2H2.^2 + eps_1;

        % wiener filter

        SPEECH1 = ((W1H1.^2)./V_ap).*X;
        SPEECH2 = ((W2H2.^2)./V_ap).*X;
        speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
        speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);

        Parms2 =  BSS_EVAL(x1', x2', speech1', speech2', mix');
        

        Parms2
        output2{i} = Parms2;
        

        %% ---
%         Old with bug!
%
%         V_ap = W1H1 +W2H2 + eps;
% 
%         % wiener filter
% 
%         SPEECH1 = ((W1H1.^2)./(V_ap.^2)).*X;
%         SPEECH2 = ((W2H2.^2)./(V_ap.^2)).*X;
%         speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
%         speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);
% 
%         Parms =  BSS_EVAL(x1', x2', speech1', speech2', mix');
%         
%         NSDR = SDR+mean(Parms.NSDR)/N_test;
%         SIR = SIR+mean(Parms.SIR)/N_test;
%         
%         Parms
        
        %%-----
        
        if 0

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

    end
    save_file = sprintf('%sresults2.mat',save_folder,'s');
    save(save_file,'output','D1','D2','param','fv','r')
    unix(sprintf('chmod 777 %s ',save_file));
    AA{ii,jj}.res = output;
    %clear output
end
end

