
root = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/TRAIN/';
root_save = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/';

% sampling parameters
fs = 16000;
NFFT = 2048;
hop = NFFT/2;

Npad = 2^15;

scparam.T = NFFT;
scparam.os = 1;%NFFT/hop;
scparam.Q = 32;
scparam.N=Npad;

filts = cqt_prepare(scparam);


label = 'cqt_phase';

save_folder_train = sprintf('%s%s_fs%d_NFFT%d_hop%d/',root_save,label,fs/1000,NFFT,hop);

unix(sprintf('mkdir %s',save_folder_train));
unix(sprintf('chmod 777 %s ',save_folder_train));

save_folder = sprintf('%sTRAIN/',save_folder_train);

unix(sprintf('mkdir %s',save_folder));
unix(sprintf('chmod 777 %s ',save_folder));


d1 = dir(root);

for i = 1:length(d1)
    
    if d1(i).isdir && ~strcmp(d1(i).name(1),'.')
    
        dialect = d1(i).name;
        
        folder = sprintf('%s%s/',root,dialect);
        
        d2 = dir(folder);
        
        for j = 1:length(d2)
    
        if d2(j).isdir && ~strcmp(d2(j).name(1),'.')
            
            speaker = d2(j).name;
            folder_speaker = sprintf('%s%s/',folder,speaker);
            files = dir(sprintf('%s/*.WAV',folder_speaker));
            
            disp('----------------------------------------------')
            fprintf('%s\n',d2(j).name)
            
            X = [];
            
            for k = 1:length(files)
                
                
                fprintf('%s\n',files(k).name)
                
                [x,Fs] = audioread(sprintf('%s%s',folder_speaker,files(k).name));
                x = resample(x,fs,Fs);
                Xo(:,k)=pad_mirror(x,Npad);

            end
            
            
            [~,~,X] = batchscatt(Xo,filts, scparam);    
            
            
            data.X = X(:,:);
            data.dialect = dialect;
            data.speaker = speaker;
            data.NFFT = NFFT;
            data.hop = hop;
            data.fs = fs;
            data.scparam = scparam;
            data.files = files;
            data.folder = folder_speaker;
            
            save_file = sprintf('%s/%s_%s.mat',save_folder,dialect,speaker);
            save(save_file,'data')
            unix(sprintf('chmod 777 %s ',save_file));
            
            clear data X
        end
        
        end
        

    end
end

