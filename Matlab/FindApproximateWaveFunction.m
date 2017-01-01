function [Psi,dPsi] = FindApproximateWaveFunction(data,coeff,sigma,x)
% finds the approximate wave function\Parzen function and its gradient.
% data - matrix with data. each row corresponds to one data point.
% coeff - a weight for wach row in data
% sigma - scalar, the parameter that appears in the Parzen wavefunction.
% x - matrix with points where the wave function will be evaluated. Each row is a point. If x is empty, then x = data.
% V - the wave function. It is a column vector with size(x,1) elements.  
% dV - the gradient of the wave function at the points x. It has the same size as x.	

	
	if isempty(x)
		x = data;
	end
	
	Psi = zeros(size(x,1),1);
	dPsi = zeros(size(x));
	for ii=1:size(x,1)
		difference = (repmat(x(ii,:),size(data,1),1) - data);
		squaredDifference = sum(difference.^2,2);
		gaussian = exp(-(1/(2*sigma^2))*squaredDifference);
		Psi(ii) = sum(coeff.*gaussian);
		
		dPsi(ii,:) = -1*sum(difference.*repmat(coeff.*gaussian,1,size(data,2))*2*sigma^2);
	end	

end