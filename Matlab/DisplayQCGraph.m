function DisplayQCGraph(x,xHistory,clusters)

	captureVideo = 1;
	
	if (~exist('clusters','var') || isempty(clusters))
		clusters = ones(size(xHistory,1),1);
	end
	
	if isempty(x)
		x = 1:size(xHistory,2);
	end
	
	nclusters = max(clusters);
	plt = zeros(size(xHistory,1),1);
	clr = linspecer(nclusters);
% 	clr = clr(randperm(nclusters),:);
	figure;	
	ax = axes;
	hold on;
	acc = accumarray(clusters,1);
	[~,inds] = sort(acc);
	for ii=1:nclusters
		plt(clusters==inds(ii)) = plot(x,xHistory(clusters==inds(ii),:,1)','Color',clr(inds(ii),:));
	end
	xlim([min(x),max(x)]);
	ylim([min(xHistory(:)),max(xHistory(:))]);
	title(['step #0/' num2str(size(xHistory,3)-1)]);

	
	if captureVideo
		frm = print('-RGBImage','-opengl','-noui');	
		vid = VideoWriter('a.mp4','MPEG-4');
		vid.FrameRate = 10;
		open(vid);
		for ii=1:6
			writeVideo(vid,frm);
		end
	end
	
	
	for ii=1:(size(xHistory,3)-1)
		for jj=1:size(xHistory,1)
			set(plt(jj),'YData',xHistory(jj,:,ii)');
		end
		title(['step #' num2str(ii-1) '/' num2str(size(xHistory,3)-1)]);
		if captureVideo
			frm = print('-RGBImage','-opengl','-noui');
			writeVideo(vid,frm);		
		end
		pause(eps);
	end
	
	if captureVideo
		for ii=1:6
			writeVideo(vid,frm);
		end	
		close(vid);
	end
	
	maxValueInCluster = zeros(1,max(clusters));
	for ii=1:nclusters
		maxValueInCluster(ii) = max(max(xHistory(clusters==ii,:,1)));
	end
	
	[~,ind] = sortrows([maxValueInCluster(clusters(:))',clusters(:), max(xHistory(:,:,1),2)]);

	figure;
	h = waterfall(x,1:size(xHistory,1),xHistory(ind,:,1),repmat(clusters(ind),1,size(xHistory,2)));
	colormap(clr);
	h.CData((end-2):end,:) = h.CData((end-5):(end-3),:); % nan
	h.CData(1:2,:) = h.CData(3:4,:); % nan
	% h.CData((end-2):end,:) = nan;
	% h.CData(1:2,:) = nan;
	% h.FaceAlpha = 1;

	ylabel('#');

	figure;	
	ax = axes;
	hold on;
	acc = accumarray(clusters,1);
	[~,inds] = sort(acc);
	if (nclusters>9)
		inds = inds((end-8):end);
	end
	minY = inf;
	maxY = -inf;
	clr = linspecer(numel(inds),'qualitative');
	display(['ommited ' num2str(max(nclusters-9,0)) ' clusters']);
	for ii=1:numel(inds)
		sizeOfCluster = acc(inds(ii));
		y = xHistory(clusters==inds(ii),:,1);
		if (size(y,1)>1)
			sd = std(y);
			y = mean(y);
		else
			sd = 0;
		end
		minY = min(minY,min(y));
		maxY = max(maxY,max(y));
		plot(x,y,'Color',clr(ii,:),'LineWidth',3.3);
		plot(x,y+sd/2,'Color',clr(ii,:),'LineWidth',0.5);
		plot(x,y-sd/2,'Color',clr(ii,:),'LineWidth',0.5);
		text(x(end)+0.03,y(end),num2str(sizeOfCluster),'BackgroundColor','none','EdgeColor','none','HorizontalAlignment','left','VerticalAlignment','middle','FontSize',9,'FontWeight','bold');
	end
	xlim([min(x),max(x)]);
	ylim([minY,maxY]);
	grid on;
	
end