function [T, S, V] = trees(datacov, options)
%this function estimates a collection of trees using hierarchical spectral clustering of the data.


%compute the spectrum
[V,S] = graphlaplacian(datacov,options);

ntrees=getoptions(options,'ntrees',4);
%hierarchical K-means on the spectrum

for n=1:ntrees
T{n} = hkmeans(V','agglomerative',options);
end


