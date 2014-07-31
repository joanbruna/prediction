function read_whole_batch(didi);

%didi='/misc/vlgscratch3/LecunGroup/bruna/grid_data';
addpath utils
addpath stft

files=dir(didi);
maxexamples=2000000;
maxlabels=Inf;
maxexperclass=Inf;

fs = 16000;
NFFT = 640;
winsize = NFFT;
hop = winsize/2;
scf = 2/3;
p = 1;

X=zeros(1+NFFT/2,maxexamples);
meta=zeros(1,maxexamples);
for d=3:size(files,1)
	if files(d).isdir == true
		rien = files(d).name;
		if strcmp(rien,'s29')
		fprintf('processing folder %s \n', files(d).name)
		inside = dir(fullfile(didi,files(d).name));
		if d-2 > maxlabels
			return;
		end
		X = 0*X;
		meta=0*meta;
		r=1;
		for dd=3:size(inside,1)
			fname=fullfile(didi,files(d).name,inside(dd).name);
			if dd-2 > maxexperclass
				continue;
			end
			if strcmp(fname(size(fname,2)-3:size(fname,2)),'.wav')
				[x, Fs]=audioread(fname);
				x = resample(x,fs,Fs);
				x = x(:); 
				xx = scf * stft(x, NFFT ,winsize, hop);
				V = abs(xx).^p;
				if r < maxexamples - size(V,2)
					X(:,r:r+size(V,2)-1) = V; 
					meta(:,r:r+size(V,2)-1)=dd-2; %label is the file name now
					r=r+size(V,2);
				else
					continue;
                		end
            		end
		end
		%save file
		Xc = X(:,1:r-1);
		met = meta(1:r-1);
		save(sprintf('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_%d/class_%s.mat',NFFT,files(d).name),'Xc','met','-v7.3');
		%cd(didi)
		end
	end
end


