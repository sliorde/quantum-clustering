function [S,dS] = FindEntropy(data,sigma,x)
% finds the entropy and its gradient for quantum clustering
% data - matrix with data. each row corresponds to one data point.
% sigma - scalar, the parameter that appears in the Parzen wavefunction.
% x - matrix with points where the potential will be evaluated. Each row is a point. If x is not given, then x = data.
% S - the entropy. It is a column vector with size(x,1) elements.  
% dS - the gradient of the entropy at the points x. It has the same size as x. 

	assert(isa(data,'double'));
	assert(isreal(data));
	assert(all(size(data)<[1e5,1e3]));
	assert(isa(sigma,'double'));
	assert(isreal(sigma));
	assert(isscalar(sigma));
	assert(isa(x,'double'));
	assert(isreal(x));
	assert(all(size(x)<[1e5,1e3]));

	if (nargin < 3)
		x = data;
	end
	
	S = zeros(size(x,1),1);
	dS = zeros(size(x));
	for ii=1:size(x,1)
		difference = (repmat(x(ii,:),size(data,1),1) - data);
		squaredDifference = sum(difference.^2,2);
		gaussian = exp(-(1/(2*sigma^2))*squaredDifference);
		laplacian = sum(gaussian.*squaredDifference); % this is not the true Laplacian, since I omit a constant additive term from the potential
		parzen = sum(gaussian);
		V = (1/(2*sigma^2))*laplacian/parzen;
		S(ii) = V + log(abs(parzen));
		
		dS(ii,:) = (1/parzen)*sum(difference.*repmat(gaussian,1,size(data,2)).*(2*sigma^2*V-repmat(squaredDifference,1,size(data,2))));
	end	

end