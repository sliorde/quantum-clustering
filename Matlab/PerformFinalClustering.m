function clusters = PerformFinalClustering(data,stepSize,th)
% final clustering of data after performing quantum clustering.
% data - matrix, the result of performing QC
% stepSize - the step size used by QC. this step size is the resolution of movement of the replicas in gradient descent, so we cluster together points that are within a distance of the step size (times 'th') from each other.
% th - we cluster together points that are within a distance of the 'stepSize*th' from each other.
% out:
% clusters - a vector with the same number of rows as 'data', such that 'clusters(ii)' is the cluster of 'data(ii,:)'. The clusters' numbers are ordered from largest to biggest, so that cluster 1 is the largest.

	if (~exist('th','var')) || (isempty(th)) || (th==0)
		th = 3;
	end
	clusters = zeros(size(data,1),1);
	ii = 1;
	c = 1;
	distances = squareform(pdist(data));
	while ~isempty(ii)
		inds = find(clusters==0);
		clusters(inds(distances(ii,inds) <= th*stepSize))= c;
		c= c+1;		
		ii = find(clusters==0,1,'first');
	end
	
	[~,inds] = sort(accumarray(clusters,1),'descend');
	[~,inds] = sort(inds);
	clusters = inds(clusters);
end