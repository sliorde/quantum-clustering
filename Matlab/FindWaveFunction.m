function [Psi,dPsi] = FindWaveFunction(data,sigma,x)
	
	if (nargin < 3)
		x = data;
	end
	
	Psi = zeros(size(x,1),1);
	dPsi = zeros(size(x));
	for ii=1:size(x,1)
		difference = (repmat(x(ii,:),size(data,1),1) - data);
		squaredDifference = sum(difference.^2,2);
		gaussian = exp(-(1/(2*sigma^2))*squaredDifference);
		Psi(ii) = sum(gaussian);
			
		
		dPsi(ii,:) = -1*sum(difference.*repmat(gaussian,1,size(data,2))*2*sigma^2);
	end	
end