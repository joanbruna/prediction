

%% load data

representation = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_fs16_NFFT1024_hop512/';

id_1 = 2;
id_2 = 11;

% another man!
%id_2 = 14;


load(sprintf('%ss%d',representation,id_1));
data1 = data;
clear data



load(sprintf('%ss%d',representation,id_2));
data2 = data;
clear data




%% train models

model = 'NMF-L2';

KK = [10,50,100,200];
LL = [0.01,0.05,0.1,0.2];

for ii = 1:length(KK)

for jj = 1:length(LL)

    param.K = KK(ii);
    param.posAlpha = 1;
    param.posD = 1;
    param.pos = 1;
    param.lambda = LL(jj);
    param.lambda2 = 0;
    param.iter = 500;


    D1 = mexTrainDL(abs(data1.X), param);

    D2 = mexTrainDL(abs(data2.X), param);


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

        % compute decomposition
        H =  full(mexLasso(abs(X),[D1,D2],param));

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

