function clusters = PerformFinalClustering(data,stepSize)
	clusters = zeros(size(data,1),1);
	ii = 1;
	c = 1;
	distances = squareform(pdist(data));
	while ~isempty(ii)
		inds = find(clusters==0);
		clusters(inds(distances(ii,inds) <= 3*stepSize))= c;
		c= c+1;		
		ii = find(clusters==0,1,'first');
	end
end