function T = trees(data, options)
%this function estimates a collection of trees using hierarchical spectral clustering of the data.


[uu,ss,vv]=svd(data',0);
bis = ss*vv';

%compute the spectrum
V = graphlaplacian(bis,options);

ntrees=getoptions(options,'ntrees',4);
%hierarchical K-means on the spectrum

for n=1:ntrees
T{n} = hkmeans(V','agglomerative',options);
end


