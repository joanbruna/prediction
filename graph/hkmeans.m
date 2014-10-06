function out = hkmeans(data, type, options)

[N,L]=size(data);
J=getoptions(options,'Jmax',4);
switch type
	case 'agglomerative'
	C{1}=data;
	for j=1:J	
		fprintf('scale %d \n',j)
		[rien, C{j+1}]=kmeansfix(C{j},L*2^(-j));
		%[rien, C{j+1}]=greedypair(C{j});
		[~,I{j+1}]=sort(rien);
	end
	out=I{end}-1;
	for j=J:-1:2
	   nout = 2*out(ceil(I{j}/2))+mod(I{j},2);
	   out=nout;
	end
	out=out+1;
	case 'divisive'
		error('not done yet')
	end
end

