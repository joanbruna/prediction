
root = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/TRAIN/';
root_save = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/';

% sampling parameters
fs = 16000;
Npad = 2^16;
T = 2048;

options.N = Npad;
options.T = T;
options.Q = 32;

filts = create_scattfilters(options);


label = 'scatt2';

save_folder_train = sprintf('%s%s_fs%d_NFFT%d/',root,label,fs/1000,T);

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
            
	[X2, X1] = audioscatt_fwd_haar(Xo,filts, options);    

	    data.X1 = X1(:,:);
	    data.X2 = X2(:,:);
	    data.T = T;
	    data.fs = fs;
	    data.filts = filts;
	    data.scparam = options;
            data.dialect = dialect;
            data.speaker = speaker;
            data.files = files;
            data.folder = folder_speaker;
            
            save_file = sprintf('%s/%s_%s.mat',save_folder,dialect,speaker);
            save(save_file,'data', '-v7.3');
            unix(sprintf('chmod 777 %s ',save_file));
            
	fprintf('done %d \n', i)
            clear data X1 X2
        end
        
        end
        

    end
end

