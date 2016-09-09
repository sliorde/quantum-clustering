data = load('Iris.csv');
reducedData = data(:,1:4);

sigma = 0.5	; % parameter of QC - width of gaussians
rep = 	500; % number of steps for gradient descent
stepSize = 0.15; % step size for gradient descent
withDisplay = 0; % should show display?
clusteringType = 'V'; % 'V' for qunatum clustering, 'S' for entropy maximization
recalculate = false; % should potential/entropy function depend on initial points or be recalculated?
normalizeData = false; % should normalize data at each step?

[x,xHistory] = PerformGDQC(reducedData,sigma,uint16(rep),stepSize,clusteringType,recalculate,normalizeData);

clusters = PerformFinalClustering(x,stepSize);

if withDisplay
	DisplayQCGraph(wl,xHistory,clusters);
end