function DisplayQC(xHistory,clusters,principalComponents,dataInPCbasis)
% displays the quantum clustering process dymanically
% xHistory - the entire evolution of QC as returned from the function PerformGDQQC
% clusters - vector with cluster indexes of data points. numel(clusters) = size(xHistory,1). can be [] for no clustering.
% optonal:principalComponents - the principal components of the original data, as returned from function pca. The display will project the data back to the original basis.
% optonal:dataInPCbasis - the original data in the   principal component basis. The display will also use the coordinated of the unused principal components in the projection.
	
	captureVideo = false;


	showLines = 0;

	if (nargin>=3)
		n = size(xHistory,2);
		newXHistory = zeros(size(xHistory,1),size(principalComponents,1),size(xHistory,3));
		for ii=1:size(xHistory,3)
			if (nargin==4)
				newXHistory(:,:,ii) = [xHistory(:,:,ii),dataInPCbasis(:,(n+1):end)]*principalComponents';
			else
				newXHistory(:,:,ii) = xHistory(:,:,ii)*principalComponents(:,1:n)';
			end
		end
		xHistory = newXHistory;
	end
	
	if isempty(clusters)
		clusters = ones(size(xHistory,1),1);
	end
	
	if (size(xHistory,2) == 1)
		figure;
		axis equal;
		clrs = myColorMap(max(clusters));
		[~,~,tmp] = unique(clusters);
		clrs = clrs(tmp,:);
		sc = scatter(xHistory(:,1,1),xHistory(:,1,1)*0,10,clrs,'filled');
		xlim([min(min(xHistory(:,1,:))) , max(max(xHistory(:,1,:)))]);
		ylim([-1 , 1]);
		if captureVideo
			frm = print('-RGBImage','-r0','-opengl','-noui');	
% 		 		frm = frm(250:570,:,:);
			frames = zeros(size(frm,1),size(frm,2),3,20);
			nextFrmInQueue = 1;
			frames(:,:,:,nextFrmInQueue) = frm;
			nextFrmInQueue = nextFrmInQueue+1;
			vid = VideoWriter('filename.mp4','MPEG-4');
			vid.FrameRate = 18;
			open(vid);
		end
		for ii=1:size(xHistory,3)
			set(sc,'XData',xHistory(:,1,ii));
			title(['step #' num2str(ii-1) '/' num2str(size(xHistory,3))]);
			if captureVideo
				frm = print('-RGBImage','-r0','-opengl','-noui');
		% 		frm = frm(72:700,180:1050,:);
				frames(:,:,:,nextFrmInQueue) = frm;
				nextFrmInQueue = nextFrmInQueue+1;
				if (nextFrmInQueue==21)
					writeVideo(vid,frames/255);
					nextFrmInQueue = 1;
				end
			end
			pause(0.05);
		end
		if captureVideo
			frames(:,:,:,nextFrmInQueue) = frm;	
			writeVideo(vid,frames(:,:,:,1:nextFrmInQueue)/255);
			close(vid);
		end		
		figure;
		axis equal;
		scatter(xHistory(:,1,1),xHistory(:,1,1)*0,10,clrs,'filled');
		hold on;
		xlim([min(min(xHistory(:,1,:))) , max(max(xHistory(:,1,:)))]);
		ylim([-1 , 1]);
% 		for ii=1:max(clusters)
% % 			text(mean(xHistory(clusters==ii,1,1)),0.5,num2str(ii),'BackgroundColor','w','EdgeColor','k','HorizontalAlignment','center','VerticalAlignment','middle');
% 			text(mean(xHistory(clusters==ii,1,1)),0.5,num2str(sum(clusters==ii)),'BackgroundColor','w','EdgeColor','k','HorizontalAlignment','center','VerticalAlignment','middle');
% 		end
	elseif (size(xHistory,2) == 2)
		figure;
		axis equal;
		clrs = myColorMap(max(clusters));
		[~,~,tmp] = unique(clusters);
		clrs = clrs(tmp,:);
		sc = scatter(xHistory(:,1,1),xHistory(:,2,1),10,clrs,'filled');
		xlim([min(min(xHistory(:,1,:))) , max(max(xHistory(:,1,:)))]);
		ylim([min(min(xHistory(:,2,:))) , max(max(xHistory(:,2,:)))]);
		if captureVideo
			frm = print('-RGBImage','-r0','-opengl','-noui');	
% 		 		frm = frm(250:570,:,:);
			frames = zeros(size(frm,1),size(frm,2),3,20);
			nextFrmInQueue = 1;
			frames(:,:,:,nextFrmInQueue) = frm;
			nextFrmInQueue = nextFrmInQueue+1;
			vid = VideoWriter('filename.mp4','MPEG-4');
			vid.FrameRate = 18;
			open(vid);
		end
		for ii=1:size(xHistory,3)
			set(sc,'XData',xHistory(:,1,ii),'YData',xHistory(:,2,ii));
			title(['step #' num2str(ii-1) '/' num2str(size(xHistory,3))]);			
			if ((ii>1) & showLines)
				line([xHistory(:,1,ii-1)';xHistory(:,1,ii)'],[xHistory(:,2,ii-1)';xHistory(:,2,ii)'],'Color',[0.95,0.95,0.95]);
			end
			uistack(sc,'top');		
			if captureVideo
				frm = print('-RGBImage','-r0','-opengl','-noui');
		% 		frm = frm(72:700,180:1050,:);
				frames(:,:,:,nextFrmInQueue) = frm;
				nextFrmInQueue = nextFrmInQueue+1;
				if (nextFrmInQueue==21)
					writeVideo(vid,frames/255);
					nextFrmInQueue = 1;
				end
			end		
			pause(0.05);
		end
		if captureVideo
			frames(:,:,:,nextFrmInQueue) = frm;	
			writeVideo(vid,frames(:,:,:,1:nextFrmInQueue)/255);
			close(vid);
		end	
		figure;
		axis equal;
		scatter(xHistory(:,1,1),xHistory(:,2,1),10,clrs,'filled');
		xlim([min(min(xHistory(:,1,:))) , max(max(xHistory(:,1,:)))]);
		ylim([min(min(xHistory(:,2,:))) , max(max(xHistory(:,2,:)))]);
% 		for ii=1:max(clusters)
% % 			text(mean(xHistory(clusters==ii,1,1)),mean(xHistory(clusters==ii,2,1)),num2str(ii),'BackgroundColor','w','EdgeColor','k','HorizontalAlignment','center','VerticalAlignment','middle');
% 			text(mean(xHistory(clusters==ii,1,1)),mean(xHistory(clusters==ii,2,1)),num2str(sum(clusters==ii)),'BackgroundColor','w','EdgeColor','k','HorizontalAlignment','center','VerticalAlignment','middle');
% 		end
	else
		if ((size(xHistory,2) > 3) && (nargin<3))
			[principalComponents,dataInPCbasis,~] = pca(xHistory(:,:,1),'Centered',false);
			n = size(xHistory,2);
			newXHistory = zeros(size(xHistory,1),size(dataInPCbasis,2),size(xHistory,3));
			for ii=1:size(xHistory,3)
				newXHistory(:,:,ii) = xHistory(:,:,ii)*principalComponents;
			end
			xHistory = newXHistory;
		end
		figure;
		axis equal;
		clrs = myColorMap(max(clusters));
		[~,~,tmp] = unique(clusters);
		clrs = clrs(tmp,:);
		sc = scatter3(xHistory(:,1,1),xHistory(:,2,1),xHistory(:,3,1),10,clrs,'filled');
		xlim([min(min(xHistory(:,1,:))) , max(max(xHistory(:,1,:)))]);
		ylim([min(min(xHistory(:,2,:))) , max(max(xHistory(:,2,:)))]);
		zlim([min(min(xHistory(:,3,:))) , max(max(xHistory(:,3,:)))]);
		if captureVideo
			frm = print('-RGBImage','-r0','-opengl','-noui');	
% 		 		frm = frm(250:570,:,:);
			frames = zeros(size(frm,1),size(frm,2),3,20);
			nextFrmInQueue = 1;
			frames(:,:,:,nextFrmInQueue) = frm;
			nextFrmInQueue = nextFrmInQueue+1;
			vid = VideoWriter('filename.mp4','MPEG-4');
			vid.FrameRate = 18;
			open(vid);
		end
		for ii=1:(size(xHistory,3)-1)
			set(sc,'XData',xHistory(:,1,ii),'YData',xHistory(:,2,ii),'ZData',xHistory(:,3,ii));
			title(['step #' num2str(ii-1) '/' num2str(size(xHistory,3)-1)]);			
			if ((ii>1) & showLines)
				line([xHistory(:,1,ii-1)';xHistory(:,1,ii)'],[xHistory(:,2,ii-1)';xHistory(:,2,ii)'],[xHistory(:,3,ii-1)';xHistory(:,3,ii)'],'Color',[0.95,0.95,0.95]);
			end
			uistack(sc,'top');
			if captureVideo
				frm = print('-RGBImage','-r0','-opengl','-noui');
		% 		frm = frm(72:700,180:1050,:);
				frames(:,:,:,nextFrmInQueue) = frm;
				nextFrmInQueue = nextFrmInQueue+1;
				if (nextFrmInQueue==21)
					writeVideo(vid,frames/255);
					nextFrmInQueue = 1;
				end
			end		
			pause(0.05);
		end
		if captureVideo
			frames(:,:,:,nextFrmInQueue) = frm;	
			writeVideo(vid,frames(:,:,:,1:nextFrmInQueue)/255);
			close(vid);
		end
		figure;
		axis equal;
		scatter3(xHistory(:,1,1),xHistory(:,2,1),xHistory(:,3,1),10,clrs,'filled');
		axis equal;
		xlim([min(min(xHistory(:,1,:))) , max(max(xHistory(:,1,:)))]);
		ylim([min(min(xHistory(:,2,:))) , max(max(xHistory(:,2,:)))]);
		zlim([min(min(xHistory(:,3,:))) , max(max(xHistory(:,3,:)))]);
		clr = jet(max(clusters));
% 		for ii=1:max(clusters)
% % 			text(mean(xHistory(clusters==ii,1,1)),mean(xHistory(clusters==ii,2,1)),mean(xHistory(clusters==ii,3,1)),num2str(ii),'BackgroundColor','w','EdgeColor','k','HorizontalAlignment','center','VerticalAlignment','middle');
% 			text(mean(xHistory(clusters==ii,1,1)),mean(xHistory(clusters==ii,2,1)),mean(xHistory(clusters==ii,3,1)),num2str(sum(clusters==ii)),'BackgroundColor','w','EdgeColor','k','HorizontalAlignment','center','VerticalAlignment','middle');
% 		end
	end
	
% 	if (size(xHistory,2) > 1)
% 		figure;
% 		clr = jet(max(clusters));
% 		hold on;
% 		for ii=1:max(clusters)
% 			plot(xHistory(clusters==ii,:,1)','Color',clr(ii,:));
% 		end
% 	end
	
end