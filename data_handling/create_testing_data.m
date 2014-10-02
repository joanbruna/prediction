

root = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/';
root_save = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/';

% sampling parameters
fs = 16000;
NFFT = 1024;
hop = NFFT/2;


label = 'spect';

save_folder_train = sprintf('%s%s_fs%d_NFFT%d_hop%d/',root_save,label,fs/1000,NFFT,hop);

unix(sprintf('mkdir %s',save_folder_train));
unix(sprintf('chmod 777 %s ',save_folder_train));

save_folder = sprintf('%sTEST/',save_folder_train);

unix(sprintf('mkdir %s',save_folder));
unix(sprintf('chmod 777 %s ',save_folder));

%label = 'female';
label = 'male';

d1 = dir(root);

test_female= cell(length(d1)-3,1);
count = 1;

for i = 1:length(d1)
    
    if d1(i).isdir && ~strcmp(d1(i).name(1),'.')
    
        dialect = d1(i).name;
        
        %folder = sprintf('%s%s/',root,dialect);
        folder = sprintf('%s%s/',root,dialect);
        
        %d2 = dir([folder 'F*']);
        d2 = dir([folder 'M*']);
        
        sp = cell(length(d2),1);
        
        for j = 1:length(d2)
    
        if d2(j).isdir && ~strcmp(d2(j).name(1),'.')
            
            
            speaker = d2(j).name;
            folder_speaker = sprintf('%s%s/',folder,speaker);
            files = dir(sprintf('%s/*.WAV',folder_speaker));
            
            disp('----------------------------------------------')
            fprintf('%s\n',d2(j).name)
            
            X = [];
            
            a.dialect = dialect;
            a.speaker = speaker;
            
            
            for k = 1:length(files)
                
                
                fprintf('%s\n',files(k).name)
                
                [x,Fs] = audioread(sprintf('%s%s',folder_speaker,files(k).name));
                x = resample(x,fs,Fs);
                x = x(:)';
                
                a.file = files(k).name;
                a.x = x;
                a.fs = fs;
                
                file{k} = a;
                
                
            end
            
            
            sp{j} = file;
            clear a
            clear file
            
            
            
        end

        end
        
        test_male{count} = sp;
        count = count +1;
        clear sp
    end
end

save_file = sprintf('%s/%s_audios.mat',save_folder,label);
save(save_file,'test_male')
unix(sprintf('chmod 777 %s ',save_file));

break


%% For Creating the MAT files

root = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/TEST/';
root_save = '/misc/vlgscratch3/LecunGroup/pablo/TIMIT/';

% sampling parameters
fs = 16000;
NFFT = 1024;
hop = NFFT/2;


label = 'spect';

save_folder_train = sprintf('%s%s_fs%d_NFFT%d_hop%d/',root_save,label,fs/1000,NFFT,hop);

unix(sprintf('mkdir %s',save_folder_train));
unix(sprintf('chmod 777 %s ',save_folder_train));

save_folder = sprintf('%sTEST/',save_folder_train);

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
                x = x(:)';
                
                Xt = compute_spectrum(x,NFFT,hop);
                
                X = [X,Xt];
            end
            
            data.X = X;
            data.dialect = dialect;
            data.speaker = speaker;
            data.NFFT = NFFT;
            data.hop = hop;
            data.fs = fs;
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

    





