function [x,xHistory] = PerformGDQC(data,sigma,rep,stepSize,clusteringType,recalculatePotential,normalizeData,dataInitialPosition,normalizeGradient,stochasticSz,voxelSize,howOftenToTestIfDone)
% performs gradient descent on data using potential from quantum clustering.
% data - matrix with data. each row corresponds to one data point.
% sigma - scalar, the parameter that appears in the Parzen wavefunction.
% rep - scalar, maximal number of steps of gradient descent. default: 200
% stepSize - scalar,  gradient will be multiplied by this number to perform each gradient descent step. default: 'sigma/7'
% clusteringType - char, either 'V','S' or 'P' for minimizing the potential, maximizing the entropy or maximizing the wavefunction. default: 'S'
% recalculatePotential - boolen. The gradient on each step will be derived from the potential using the current replica points (not initial data points). default: false
% normalizeData - boolean, Each data point will be normalized to norm 1 on each gradient descent step. default: false
% dataInitialPosition - matrix with same column number as 'data'. Each row is a point to be moved by the gradients. If empty, 'data' will be used. default: 'data'
% normalizeGradient - boolean. The gradient is normalized to unit norm before multiplied by 'stepSize'. default: 'data'
% stochasticSz - scalar. If this is non-empty, a stochastic version of the algorithm will be used, in which the gradient is calculated based on a random sample of 'stochasticSz*size(data,1)' of the data points. default: []
% howOftenToTestIfDone - scalar. once in how many steps to check wheter a point is done moving. This is done by comparing the current value of the potential\entropy\wave function to the previous value 'howOftenToTestIfDonce' steps ago. default:1
% out:
% x - the replica points after gradient descent evolution.
% optional output: xHistory - a 3d matrix with size [size(x,1),size(x,2),rep]. xHistory(:,:,ii) contains the points after the (ii-1)'th step of gradient descent.


	if ~exist('stepSize','var') || isempty(stepSize)
		stepSize = sigma/7;
	end
	
	if ~exist('rep','var') || isempty(rep)
		rep = 200;
	end
	
	if ~exist('clusteringType','var') || isempty(clusteringType)
		clusteringType = 'V';
	end
	
	if ~exist('recalculatePotential','var') || isempty(recalculatePotential)
		recalculatePotential = false;
	end
	
	if ~exist('normalizeData','var') || isempty(normalizeData)
		normalizeData = false;
	end
	
	if ~exist('dataInitialPosition','var') || isempty(dataInitialPosition)
		dataInitialPosition = data;
	end
	
	if ~exist('normalizeGradient','var') || isempty(normalizeGradient)
		normalizeGradient = true;		
	end
	
	if ~exist('stochasticSz','var')
		stochasticSz = [];		
	end
	
	if ~exist('voxelSize','var')
		voxelSize = [];		
	end
	
	if ~exist('howOftenToTestIfDone','var') || isempty(howOftenToTestIfDone)
		howOftenToTestIfDone = 1;		
	end
		

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
	
	if ~isempty(voxelSize)
		[data,coeff] = getApproximateWaveFunction(data,sigma,voxelSize);
	end
	
	x = dataInitialPosition;
	
	if normalizeData
		data = normr(data);
		x = normr(x);
	end
	
	if (nargout>=2)
		xHistory = zeros(size(x,1),size(x,2),rep+1);
		xHistory(:,:,1) = x;
	end
	
	% initialize preV to have the previous values of V\S\Psi, so that can be compared to current values on each step of algorithm to check if arrived at extremum 
	prevV = inf(size(x,1),1);
	
	% these are the indices of replicas that are currently sill moving
	inds = 1:size(x,1);
	
	% master loop
	for ii=1:rep
		ii
		if isempty(voxelSize)
			if recalculatePotential
				if maximizeEntropy
					if isempty(stochasticSz)
						[S,dx] = FindEntropy(x,sigma,x);
					else
						[S,dx] = FindEntropyStochastic(x,sigma,x,stochasticSz);
					end
					if normalizeGradient
						dx = normr(dx);
					end
					x = x + stepSize*dx;
					V = -1*S;
				elseif maximizeWaveFunction
					if isempty(stochasticSz)
						[P,dx] = FindWaveFunction(x,sigma,x);
					else
						[P,dx] = FindWaveFunctionStochastic(x,sigma,x,stochasticSz);
					end
					if normalizeGradient
						dx = normr(dx);
					end
					x = x + stepSize*dx;
					V = -1*P;
				else
					if isempty(stochasticSz)
						[V,dx] = FindPotential(x,sigma,x);	
					else
						[V,dx] = FindPotentialStochastic(x,sigma,x,stochasticSz);
					end
					if normalizeGradient
						dx = normr(dx);
					end
					x = x - stepSize*dx;
					V = V;
				end
			else
				if maximizeEntropy
					if isempty(stochasticSz)
						[S,dx] = FindEntropy(data,sigma,x(inds,:));
					else
						[S,dx] = FindEntropyStochastic(data,sigma,x(inds,:),stochasticSz);
					end
					if normalizeGradient
						dx = normr(dx);
					end
					x(inds,:) = x(inds,:) + stepSize*dx;
					V = -1*S;
				elseif maximizeWaveFunction
					if isempty(stochasticSz)
						[P,dx] = FindWaveFunction(data,sigma,x(inds,:));
					else
						[P,dx] = FindWaveFunctionStochastic(data,sigma,x(inds,:),stochasticSz);
					end
					if normalizeGradient
						dx = normr(dx);
					end
					x(inds,:) = x(inds,:) + stepSize*dx;
					V = -1*P;
				else
					if isempty(stochasticSz)
						[V,dx] = FindPotential(data,sigma,x(inds,:));
					else
						[V,dx] = FindPotentialStochastic(data,sigma,x(inds,:),stochasticSz);
					end
					if normalizeGradient
						dx = normr(dx);
					end
					x(inds,:) = x(inds,:) -  stepSize*dx;
					V = V;
				end
			end
		else
			if recalculatePotential
				if maximizeEntropy
					if isempty(stochasticSz)
						[S,dx] = FindApproximateEntropy(x,coeff,sigma,[x;data]);
					else
						[S,dx] = FindApproximateStochasticEntropy(x,coeff,sigma,[x;data],stochasticSz);
					end
					if normalizeGradient
						dx = normr(dx);
					end
					x = x + stepSize*dx;
					V = -1*S;
				elseif maximizeWaveFunction
					if isempty(stochasticSz)
						[P,dx] = FindApproximateWaveFunction(x,coeff,sigma,[x;data]);
					else
						[P,dx] = FindApproximateStochasticWaveFunction(x,coeff,sigma,[x;data],stochasticSz);
					end
					if normalizeGradient
						dx = normr(dx);
					end
					x = x + stepSize*dx;
					V = -1*P;
				else
					if isempty(stochasticSz)
						[V,dx] = FindApproximatePotential(x,coeff,sigma,[x;data]);
					else
						[V,dx] = FindApproximateStochasticPotential(x,coeff,sigma,[x;data],stochasticSz);
					end
					if normalizeGradient
						dx = normr(dx);
					end
					x = x - stepSize*dx;
					V = V;
				end
			else
				if maximizeEntropy
					if isempty(stochasticSz)
						[S,dx] = FindApproximateEntropy(data,coeff,sigma,x(inds,:));
					else
						[S,dx] = FindApproximateStochasticEntropy(data,coeff,sigma,x(inds,:),stochasticSz);
					end
					if normalizeGradient
						dx = normr(dx);
					end
					x(inds,:) = x(inds,:) + stepSize*dx;
					V = -1*S;
				elseif maximizeWaveFunction
					if isempty(stochasticSz)
						[P,dx] = FindApproximateWaveFunction(data,coeff,sigma,x(inds,:));
					else
						[P,dx] = FindApproximateStochasticWaveFunction(data,coeff,sigma,x(inds,:),stochasticSz);
					end
					if normalizeGradient
						dx = normr(dx);
					end
					x(inds,:) = x(inds,:) + stepSize*dx;
					V = -1*P;
				else
					if isempty(stochasticSz)
						[V,dx] = FindApproximatePotential(data,coeff,sigma,x(inds,:));
					else
						[V,dx] = FindApproximateStochasticPotential(data,coeff,sigma,x(inds,:),stochasticSz);
					end
					if normalizeGradient
						dx = normr(dx);
					end
					x(inds,:) = x(inds,:) -  stepSize*dx;
					V = V;
				end
			end			
		end
		if normalizeData
			x(inds,:) = normr(x(inds,:));
		end

		if (nargout>=2)
			xHistory(:,:,ii+1) = x;
		end	
		
		% check if points can stop. This condition makes sense only when V\S\Psi are fixed, not changing with time as in the case of recalculatePotential=true.
		if (~recalculatePotential) && (mod(ii,howOftenToTestIfDone)==0)
			prevInds = inds;
			inds = inds(prevV(inds)>V);
			if isempty(inds)
				break;
			end
% 			inds = 1:size(x,1);
			prevV(prevInds) = V;
		end
	end
	if (nargout>=2)
		xHistory = xHistory(:,:,1:(ii+1));
	end
end