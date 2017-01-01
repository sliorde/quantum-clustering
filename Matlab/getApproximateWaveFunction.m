function [newData,coeff] = getApproximateWaveFunction(data,sigma,voxelSize)
% finds one representative data point for each voxel, and assigns a weight to each such data point.
% data - matrix with data. each row corresponds to one data point.
% sigma - scalar, the parameter that appears in the Parzen wavefunction.
% voxelSize - scalar, the size of the size of one voxel.
% out:
% newData - marix. A new set of data points, with only one data point per voxel.
% coeff - vector, same size as size(newData,1). The weight\coefficient of the corresponding new data point.

	newData = unique(floor(data/voxelSize)*voxelSize+voxelSize/2,'rows');

 	N = zeros(numel(newData));
	N = squareform(pdist(newData)).^2;
	N = -N;
	N = exp(N/(4*sigma^2));
	
	M = zeros(numel(newData),numel(data));
	M = pdist2(newData,data).^2;
	M = -M;
	M = exp(M/(4*sigma^2));
	
	C = N\M;
	
	coeff = sum(C,2);
	
	coeff = size(data,1)*coeff/sum(coeff);

end