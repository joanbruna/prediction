function [X,meta] = read_whole_batch(didi);

%didi='/misc/vlgscratch3/LecunGroup/bruna/grid_data';

files=dir(didi);
maxexamples=8000000;
maxlabels=33;
maxexperclass=15;

fs = 16000;
NFFT = 640;
winsize = NFFT;
hop = winsize/2;
scf = 2/3;
p = 1;

X=zeros(1+NFFT/2,maxexamples);
meta=zeros(1,maxexamples);
i=1;
r=1;
for d=3:size(files,1)
	if files(d).isdir == true
		fprintf('processing folder %s \n', files(d).name)
		%cd(fullfile(didi,files(d).name))
		inside = dir(fullfile(didi,files(d).name));
		if d-2 > maxlabels
			return;
		end
		for dd=3:size(inside,1)
			fname=fullfile(didi,files(d).name,inside(dd).name);
			if dd-2 > maxexperclass
				continue;
			end
			if strcmp(fname(size(fname,2)-3:size(fname,2)),'.wav')
				[x, Fs]=audioread(fname);
				x = resample(x,Fs,fs);
				x = x(:); 
				xx = scf * stft(x, NFFT ,winsize, hop);
				V = abs(xx).^p;
				if r < maxexamples - size(V,2)
					X(:,r:r+size(V,2)-1) = V; 
					meta(:,r:r+size(V,2)-1)=d; %label is the folder
					r=r+size(V,2);
					i=i+1;
				else
					return;
                		end
            		end
		end
		%cd(didi)
	end
end


