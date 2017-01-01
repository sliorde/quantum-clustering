function [Psi,dPsi] = FindApproximateStochasticWaveFunction(data,coeff,sigma,x,sz)
% finds the estimated approximate wave function\Parzen function and its gradient.
% data - matrix with data. each row corresponds to one data point.
% coeff - a weight for wach row in data
% sigma - scalar, the parameter that appears in the Parzen wavefunction.
% x - matrix with points where the wave function will be evaluated. Each row is a point. If x is empty, then x = data.
% sz - a number between 0 and 1. The wavefunction and gradient will be calculated based on a random sample of size 'sz*size(data,1)' of data points
% V - the wave function. It is a column vector with size(x,1) elements.  
% dV - the gradient of the wave function at the points x. It has the same size as x.	

	
	if isempty(x)
		x = data;
	end
	sz = ceil(sz*size(data,1));
	
	Psi = zeros(size(x,1),1);
	dPsi = zeros(size(x));
	for ii=1:size(x,1)
		inds = randi(size(data,1),sz,1);
		currentData = data(inds,:);
		
		difference = (repmat(x(ii,:),size(currentData,1),1) - currentData);
		squaredDifference = sum(difference.^2,2);
		gaussian = exp(-(1/(2*sigma^2))*squaredDifference);
		Psi(ii) = sum(coeff.*gaussian);
		
		dPsi(ii,:) = -1*sum(difference.*repmat(coeff.*gaussian,1,size(currentData,2))*2*sigma^2);
	end	

end