function [V,dV] = FindApproximatePotential(data,coeff,sigma,x)
% finds the approximate potenatial and its gradient for quantum clustering
% data - matrix with data. each row corresponds to one data point.
% coeff - a weight for wach row in data
% sigma - scalar, the parameter that appears in the Parzen wavefunction.
% x - matrix with points where the potential will be evaluated. Each row is a point. If x is empty, then x = data.
% V - the potential that gives the Parzen wavefunction as an eigenfunction. It is a column vector with size(x,1) elements.  
% dV - the gradient of the potential at the points x. It has the same size as x. 

	if isempty(x)
		x = data;
	end
	
	V = zeros(size(x,1),1);
	dV = zeros(size(x));
	for ii=1:size(x,1)
		difference = (repmat(x(ii,:),size(data,1),1) - data);
		squaredDifference = sum(difference.^2,2);
		gaussian = exp(-(1/(2*sigma^2))*squaredDifference);
		laplacian = sum(coeff.*gaussian.*squaredDifference); % this is not the true Laplacian, since I omit a constant additive term from the potential
		parzen = sum(coeff.*gaussian);
		V(ii) = 1+(1/(2*sigma^2))*laplacian/parzen;
		
		dV(ii,:) = (1/parzen)*sum(difference.*repmat(coeff.*gaussian,1,size(data,2)).*(2*sigma^2*V(ii)-repmat(squaredDifference,1,size(data,2))));
	end	
	
	V = V-1;
end