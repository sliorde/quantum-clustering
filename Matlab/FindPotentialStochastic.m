function [V,dV] = FindPotentialStochastic(data,sigma,x,sz)
% finds the estimated potenatial and its gradient for quantum clustering
% data - matrix with data. each row corresponds to one data point.
% sigma - scalar, the parameter that appears in the Parzen wavefunction.
% x - matrix with points where the potential will be evaluated. Each row is a point. If x is empty, then x = data.
% sz - a number between 0 and 1. The potential and gradient will be calculated based on a random sample of size 'sz*size(data,1)' of data points
% V - the potential that gives the Parzen wavefunction as an eigenfunction. It is a column vector with size(x,1) elements.  
% dV - the gradient of the potential at the points x. It has the same size as x. 

	if isempty(x)
		x = data;
	end
	sz = round(sz*size(data,1));

	
	V = zeros(size(x,1),1);
	dV = zeros(size(x));
	for ii=1:size(x,1)
		inds = randi(size(data,1),sz,1);
		currentData = data(inds,:);
		
		difference = (repmat(x(ii,:),size(currentData,1),1) - currentData);
		squaredDifference = sum(difference.^2,2);
		gaussian = exp(-(1/(2*sigma^2))*squaredDifference);
		laplacian = sum(gaussian.*squaredDifference); % this is not the true Laplacian, since I omit a constant additive term from the potential
		parzen = sum(gaussian);
		V(ii) = 1+(1/(2*sigma^2))*laplacian/parzen;
		
		dV(ii,:) = (1/parzen)*sum(difference.*repmat(gaussian,1,size(currentData,2)).*(2*sigma^2*V(ii)-repmat(squaredDifference,1,size(currentData,2))));
	end	
	
	V = V-1;
end