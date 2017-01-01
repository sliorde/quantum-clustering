function [S,dS] = FindEntropyStochastic(data,sigma,x,sz)
% finds the estimation for the entropy and its gradient for quantum clustering
% data - matrix with data. each row corresponds to one data point.
% sigma - scalar, the parameter that appears in the Parzen wavefunction.
% x - matrix with points where the potential will be evaluated. Each row is a point. If x is empty, then x = data.
% sz - a number between 0 and 1. The entropy and gradient will be calculated based on a random sample of size 'sz*size(data,1)' of data points
% S - the entropy. It is a column vector with size(x,1) elements.  
% dS - the gradient of the entropy at the points x. It has the same size as x. 


	if isempty(x)
		x = data;
	end
	sz = round(sz*size(data,1));
	
	S = zeros(size(x,1),1);
	dS = zeros(size(x));
	for ii=1:size(x,1)
		inds = randi(size(data,1),sz,1);
		currentData = data(inds,:);
		
		difference = (repmat(x(ii,:),size(currentData,1),1) - currentData);
		squaredDifference = sum(difference.^2,2);
		gaussian = exp(-(1/(2*sigma^2))*squaredDifference);
		laplacian = sum(gaussian.*squaredDifference); % this is not the true Laplacian, since I omit a constant additive term from the potential
		parzen = sum(gaussian);
		V = (1/(2*sigma^2))*laplacian/parzen;
		S(ii) = V + log(abs(parzen));
		
		dS(ii,:) = (1/parzen)*sum(difference.*repmat(gaussian,1,size(currentData,2)).*(2*sigma^2*V-repmat(squaredDifference,1,size(currentData,2))));
	end	

end