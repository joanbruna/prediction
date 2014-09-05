function T = trees(data, options)
%this function estimates a collection of trees using hierarchical spectral clustering of the data.

%compute the spectrum
V = graphlaplacian(data',options);


ntrees=getoptions(options,'ntrees',4);
%hierarchical K-means on the spectrum

for n=1:ntrees

kmeansfix(V,



end
