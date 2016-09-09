function [x,xHistory,sigma] = PerformGDQC(data,sigma,rep,stepSize,clusteringType,recalculatePotential,normalizeData)
% performs gradient descent on data using potential from quantum clustering
% data - matrix with data. each row corresponds to one data point.
% sigma - scalar, the parameter that appears in the Parzen wavefunction.
% rep - scalar, number of steps of gradient descent
% stepSize - scalar, gradsient will be multiplied by this number to perform each gradient descent step
% clusteringType - 'V' for qunatum clustering, 'S' for entropy maximization, 'P' for parzen maximization
% recalculatePotential - true if the gradient on each step will be derived from the potential using the current data points (not initial data points)
% normalizeData - true if  each data point should be normalized to length 1 on each gradient descent step
% x - the data points after gradient descent evolution. matrix same size as data.
% optional output: xHistory - a 3d matrix with size [size(data,1),size(data,2),rep]. xHistory(:,:,ii) contains the points after the ii-1 step of gradient descent

	maximizeEntropy = false;
	maximizeWaveFunction = false;	
	
	switch clusteringType
		case 'V'
			maximizeEntropy = false;
			maximizeWaveFunction = false;			
		case 'S'
			maximizeEntropy = true;
			maximizeWaveFunction = false;			
		case 'P'
			maximizeEntropy = false;
			maximizeWaveFunction = true;			
	end
	
	if normalizeData
		data = normr(data);
	end
	
	x = data;
	if (nargout>=2)
		xHistory = zeros(size(x,1),size(x,2),rep+1);
		xHistory(:,:,1) = x;
	end
	
	prevX = x;
	
	for ii=1:rep
		if ((ii>1) && (mod(ii,10)==1))
			if all(sum((x-prevX).^2,2) < (3*stepSize^2))
				break;
			end
			prevX = x;
		end
		if recalculatePotential
			if maximizeEntropy
				[S,dx] = FindEntropy(x,sigma,x);
				dx = normr(dx);	
				x = x + stepSize*dx;
			elseif maximizeWaveFunction
				[P,dx] = FindWaveFunction(x,sigma,x);
				dx = normr(dx);	
				x = x + stepSize*dx;
			else
				[V,dx] = FindPotential(x,sigma,x);
				dx = normr(dx);	
				x = x - stepSize*dx;
			end
		else
			if maximizeEntropy
				[S,dx] = FindEntropy(data,sigma,x);
				dx = normr(dx);	
				x = x + stepSize*dx;
			elseif maximizeWaveFunction
				[P,dx] = FindWaveFunction(data,sigma,x);
				dx = normr(dx);	
				x = x + stepSize*dx;
			else
				[V,dx] = FindPotential(data,sigma,x);
				dx = normr(dx);	
				x = x -  stepSize*dx;
			end
		end
		
		if normalizeData
			x = normr(x);
		end

		if (nargout>=2)
			xHistory(:,:,ii+1) = x;
		end	
		
	end
	if (nargout>=2)
		xHistory = xHistory(:,:,1:(ii+1));
	end
end