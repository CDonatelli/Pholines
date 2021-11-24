function sOut = getMidlines_skelClick(videoName, varargin)

%% declare midline variables for saving
sOut = struct;
Lines(1).Frame = [];
Lines(1).MidLine = [];

%% Read in video and get video paramaters
bwVideo = VideoReader(videoName);
    FrNum = bwVideo.NumberOfFrames; 
    format = bwVideo.VideoFormat;
    height=bwVideo.Height;
    width=bwVideo.Width;

%% ask a few questions about how you want to process the video
answer = questdlg('Do you have a video data struct file?', ...
    'Data Structure?', ...
    'Yes','No','No');
    switch answer
        case 'Yes'
            sOut = varargin{1};
        case 'No'
            prompt = {'Fish ID:', 'Trial ID:', 'Fish Length (mm):','Fish Weight (g):', 'Video Frame Rate:'};
            dlgtitle = 'Fish Data: Leave defaults if unknown';
            dims = [1 35];
            definput = {'n/a','n/a','0','0','30'};
            fishData = inputdlg(prompt,dlgtitle,dims,definput);
                sOut.fishID = fishData{1};
                sOut.trialID = fishData{2};
                sOut.fishLength = str2num(fishData{3});
                sOut.fishWeight = str2num(fishData{4});
                sOut.binaryFrameRate = str2num(fishData{5});
                sOut.format = bwVideo.VideoFormat;
                sOut.croppedRes = [bwVideo.Height, bwVideo.Width];
    end

    answer = questdlg('Do you want to process all frames?', ...
    'Digitized Points', ...
    'Yes','No','No');
    switch answer
        case 'Yes'
            startFrame = 1;
            endFrame = FrNum;
            skipRate = 1;
        case 'No'
            prompt = {'Enter start frame:','Enter end frame:','Enter Skip Rate'};
            dlgtitle = 'Input';
            dims = [1 35];
            definput = {'1',num2str(FrNum),'2'};
            answer = inputdlg(prompt,dlgtitle,dims,definput);
                startFrame = str2num(answer{1});
                endFrame = str2num(answer{2});
                skipRate = str2num(answer{3});
    end
    
% Number of points along the midline, change for more
    npts = 21;
    
    sOut.startFrame = startFrame;
    sOut.endFrame = endFrame;
    sOut.skipRate = skipRate;
    
    RawImage = read(bwVideo,startFrame);

%% Go through frame by frame and get the midlines
framesToProcess = startFrame:skipRate:endFrame;
    progressbar
    visual = figure();
for Index = 1:length(framesToProcess)
 
%% Read in frames
%     %Read in frame
%     if ~exist('rect','var')
        RawImage = read(bwVideo,framesToProcess(Index));
        BinaryImage = imbinarize(RawImage, 0.5);

%     else
%         RawImage = read(bwVideo,Index);
%         RawImage = imcrop(RawImage, rect);
%     end
            
%% Plot the nose and tail from DLT digitized points
%     [fishPixelsY, fishPixelsX] = find(BinaryImage);
%     [skeletonY, skeletonX] = find(Skeleton);
%     Nose = DLTpts(framesToProcess(Index),1:2);
%     Tail = DLTpts(framesToProcess(Index),3:4);
    
%% Get the midline between the nose and tail 
    %%%START KEEGAN PLAY %%%
    if framesToProcess(Index) == startFrame
        prevImage = [];
    else
        prevImage = read(bwVideo,framesToProcess(Index-1));
        prevImage = imbinarize(RawImage, 0.5);
    end 
    
    fishScale = min([height, width]);
%     Nose = DLTpts(framesToProcess(Index),1:2);
%     Tail = DLTpts(framesToProcess(Index),3:4);
    imshow(BinaryImage);
    NoseTail = ginput(2);
    Nose = NoseTail(1,:); Tail = NoseTail(2,:);
%     [inner, fishMask] = findFish(BinaryImage,imcomplement(BinaryImage), prevImage,...
%         fishScale/10,fishScale*10,framesToProcess(Index),startFrame, Nose);
    [inner, fishMask] = findFish(BinaryImage,Nose);
    
 %% Plot the nose and tail from DLT digitized points
    %fishMask = imdilate(fishMask,strel('square',4));
    Skeleton = bwmorph(fishMask,'skel',Inf);
    Skeleton = bwmorph(Skeleton,'spur',7);
    %Skeleton2 = bwmorph(fishMask2,'skel',Inf);
    
    [fishPixelsY, fishPixelsX] = find(fishMask);
    [skeletonY, skeletonX] = find(Skeleton);
    
    [frame_rows, frame_cols] = size(fishMask);
    SkelBranches = bwmorph(Skeleton,'branchpoints');
    SkelEnds = bwmorph(Skeleton,'endpoints');
    branches_and_ends = imadd(SkelBranches,SkelEnds); %add the two together
    
 %% Fix Circle Fish
    if inner~=0 %if the fish is a circle/donut shape
        tempfig=figure();
        [tstR,tstC]=find(Skeleton==1);
        clickImage = repmat(fishMask*255,[1,1,3]);
        for i=1:size(Skeleton,1)
            for j=1:size(Skeleton,2)
                if Skeleton(i,j)==1
                    clickImage(i,j,1)=255;
                    clickImage(i,j,2:3)=0;
                end
            end
        end
        imshow(clickImage)
        w=0;
        while w ~= 1
            title('ZOOM IN TO THE CONNECTED PART OF YOUR SKELETON. ENTER WHEN DONE')
            zoom on
            w=waitforbuttonpress;
        end
        zoom off
        title({'CLICK POINT TO SEPARATE SKELETON (Shown in RED).','ENTER WHEN DONE'})
        [col,rw]=ginput(1);
        [skel_row,skel_col] = find(Skeleton(:,:)>0);
        [~, ind] = euclidist([skel_row,skel_col],rw,col,1,1);
        Skeleton(skel_row(ind)-2:skel_row(ind)+2,skel_col(ind)-2:skel_col(ind)+2)=0;
        close(tempfig)
        
        clear SkelBranches SkelEnds branches_and_ends
        SkelBranches = bwmorph(Skeleton,'branchpoints');
        SkelEnds = bwmorph(Skeleton,'endpoints');
        branches_and_ends = imadd(SkelBranches,SkelEnds); %add the two together

        [path,skeletonTail,skeletonNose,p_length] = traceFish(Skeleton,branches_and_ends,SkelBranches,Nose,Tail,frame_rows,frame_cols,thresh); %find path along skeleton
        [p_row,p_col] = find(path(:,:)>0);

        if p_length < p_thresh  %if the path is shorter than the cut-off
            [path,skeletonTail,skeletonNose,p_length] = nearCircle(Skeleton,...
                Tail,SkelBranches,frame_rows,frame_cols,skeletonTail,skeletonNose,p_thresh); %feed into nearCicle to fix it
            clear p_row p_col
            [p_row,p_col] = find(path(:,:)>0);  
        end
    else %if the fish isn't a circle
        %Set reasonable threshold for distance of branch point from nose point.
        if framesToProcess(Index) == startFrame %if on start frame
            [B,~,~,~] = bwboundaries(fishMask);
            threshshort = round(size(B{1},1)/30); %base cut-off on length of fish boundary
            threshlong = threshshort*3;
            clear B
        else %if not on start frame
            fishLength = arclength(Lines(Index-1).OrderedPts(:,1), Lines(Index-1).OrderedPts(:,2));
            threshshort = round(fishLength*0.03); % 0.075 base cut-off on length of fish midline in previous frame. 7.5% since this is ~ half the length of the skull.
            threshlong = round(fishLength*0.15); % 0.075 base cut-off on length of fish midline in previous frame. 7.5% since this is ~ half the length of the skull.
        end
        
        [path,skeletonTail,skeletonNose,p_length] = traceFish(Skeleton,...
            branches_and_ends,SkelBranches,Nose,Tail,frame_rows,frame_cols,threshshort,threshlong); %find path along skeleton
        [p_row,p_col] = find(path(:,:)>0);
        
        if framesToProcess(Index) == startFrame %if on start frame
            [B,~,~,~] = bwboundaries(fishMask);
            p_thresh = round(size(B{1},1)/20); %base cut-off on length of fish boundary
            p_minlen = round(size(B{1},1)/4);
        else %if not on start frame
            p_thresh = size(Lines(Index-1).MidLine(:,1),1)/4; %base cut-off on length of fish midline in previous frame
            p_minlen = size(Lines(Index-1).MidLine(:,1),1)*0.75;
        end
        if p_length < p_thresh  %if the path is shorter than the cut-off
            [path,skeletonTail,skeletonNose,p_length] = nearCircle(Skeleton,Tail,...
                SkelEnds,frame_rows,frame_cols,skeletonTail,skeletonNose,p_minlen); %feed into nearCicle to fix it
            clear p_row p_col
            [p_row,p_col] = find(path(:,:)>0);
            fprintf('\n Frame %i executed nearCircle \n',framesToProcess(Index)); %let user know this was the case
        end
    end
    %%% END KEEGAN PLAY %%%%    
    
    %Join the nose and tail with a straight line to minimize the effect of
    %sgolay later on.
    addlines=figure('Visible','off');
    addax = axes('Parent',addlines);
    imshow(path); hold on;
    tail_join = drawline(addax,'Position',[Tail;skeletonTail]); %use straight line to join  path to DLT tail point
    nose_join = drawline(addax,'Position',[Nose;skeletonNose]); %use straight line to join  path to DLT head point
    tail_join_mask = createMask(tail_join); %create mask of straight line
    nose_join_mask = createMask(nose_join); %create mask of straight line
    interim = imadd(path, tail_join_mask); %add line to the paths mask
    interim = logical(interim); %ensure logical matrix
    path = imadd(interim, nose_join_mask); %add line to the paths mask
    path = logical(path); %ensure logical matrix
    hold off
    delete(addlines)
    [p_row,p_col] = find(path(:,:)>0);

% Plot the path to let the user check if its OK
    p_col = [Nose(1); p_col; Tail(1)]; p_row = [Nose(2); p_row; Tail(2)];
    figure(visual)
    imshow(fishMask)
    hold on
    plot(Nose(1), Nose(2),'g*', Tail(1), Tail(2), 'g*');
    plot(p_col, p_row, 'b.');

%% If there was a circle fish, let user check that generated path looks OK
    if inner~=0
        circlecheck = questdlg('Does the path look pretty good?', ...
            'Circle Check', ...
            'Yes','No','Yes');     
        switch circlecheck
            case 'Yes'
                %If the path is mostly OK, allow filling in tail if needed
                strght_ln_dist = sqrt((skeletonTail(1)-Tail(1))^2 + (skeletonTail(2)-Tail(2))^2);
                if strght_ln_dist > p_length/5 %if this distance is large relative to the length of the path
                    tempfig=figure();
                    clear p_row p_col
                    [path] = clickJoin(path,fishMask,Nose,Tail); %allow manual input to join path to DLT tail point
                    [p_row,p_col] = find(path(:,:)>0);
                    close(tempfig)
                end
            case 'No'
                %Allow user to click circle if can't auto do it.
                tempfig=figure();
                clear path p_row p_col
                [path] = clickCircle(fishMask,Nose,Tail,frame_rows,frame_cols); %function to allow user to click points along the midline, which are joined by straight lines to form path
                [p_row,p_col] = find(path(:,:)>0);
                close(tempfig)
        end
    end

%% Add midline data to data structure
    p_col = [Nose(1); p_col; Tail(1)]; p_row = [Nose(2); p_row; Tail(2)];
    temppts.XData=p_col;
    temppts.YData=p_row;

    X = p_col(1,1); Y = p_row(1,1);
 
    Lines(Index).Frame=framesToProcess(Index);       %save data in the output structure
    Lines(Index).MidLine=[p_row,p_col];
    Lines(Index).path = path;
    hold off
    
    clear p_col p_row path
    
    [Lines(Index).OrderedPts] = orderPts(Lines(Index).MidLine,Lines(Index).MidLine(1,:));
    
    progressbar(Index/length(framesToProcess))
end

%Order the midline points from nost to tail
% progressbar('Ordering Points. Please wait...')
% for i=1:size(Lines,2)
%     [Lines(i).OrderedPts] = orderPts(Lines(i).MidLine,Lines(i).MidLine(1,:));
%     progressbar(i/size(Lines,2))
% end

nfr = size(Lines,2);
x = []; y = [];
for i = 1:nfr
    %xPts = smooth(Lines(i).MidLine(:,2), 13, 'sgolay'); 
    %yPts = smooth(Lines(i).MidLine(:,1), 13, 'sgolay');
    %xPts = smooth(Lines(i).OrderedPts(:,2), 13, 'sgolay');
    %yPts = smooth(Lines(i).OrderedPts(:,1), 13, 'sgolay');
    xPts = sgolayfilt(Lines(i).OrderedPts(:,2), 2, 13);
    yPts = sgolayfilt(Lines(i).OrderedPts(:,1), 2, 13);
    randPts = rand(1,length(xPts))/1000; xPts = xPts+randPts';
    % Generate equation if the midline
    [pts, deriv, funct] = interparc(npts, xPts, yPts, 'spline');
    % add those points to an array
    x = [x,pts(:,1)]; y = [y,pts(:,2)];
end
sOut.X = x; sOut.Y = y;
% total = nfr/sOut.binaryFrameRate; 
% sOut.t = linspace(0,total,nfr)';

%% Plot the midline results and save the data
close all   %close the image 
hold on %see multiple traces
plot(x,y)
axis equal

sOut.midLines = Lines;
midlineStruct = sOut;
save([videoName,'_midlines.mat'], 'midlineStruct','-v7.3');
 

%% Functions

function [arclen,seglen] = arclength(px,py,varargin)
    % Author: John D'Errico
    % e-mail: woodchips@rochester.rr.com
    % Release: 1.0
    % Release date: 3/10/2010

    % unpack the arguments and check for errors
    if nargin < 2
      error('ARCLENGTH:insufficientarguments', ...
        'at least px and py must be supplied')
    end

    n = length(px);
    % are px and py both vectors of the same length?
    if ~isvector(px) || ~isvector(py) || (length(py) ~= n)
      error('ARCLENGTH:improperpxorpy', ...
        'px and py must be vectors of the same length')
    elseif n < 2
      error('ARCLENGTH:improperpxorpy', ...
        'px and py must be vectors of length at least 2')
    end

    % compile the curve into one array
    data = [px(:),py(:)];

    % defaults for method and tol
    method = 'linear';

    % which other arguments are included in varargin?
    if numel(varargin) > 0
      % at least one other argument was supplied
      for i = 1:numel(varargin)
        arg = varargin{i};
        if ischar(arg)
          % it must be the method
          validmethods = {'linear' 'pchip' 'spline'};
          ind = strmatch(lower(arg),validmethods);
          if isempty(ind) || (length(ind) > 1)
            error('ARCLENGTH:invalidmethod', ...
              'Invalid method indicated. Only ''linear'',''pchip'',''spline'' allowed.')
          end
          method = validmethods{ind};

        else
          % it must be pz, defining a space curve in higher dimensions
          if numel(arg) ~= n
            error('ARCLENGTH:inconsistentpz', ...
              'pz was supplied, but is inconsistent in size with px and py')
          end

          % expand the data array to be a 3-d space curve
          data = [data,arg(:)]; %#ok
        end
      end

    end

    % what dimension do we live in?
    nd = size(data,2);

    % compute the chordal linear arclengths
    seglen = sqrt(sum(diff(data,[],1).^2,2));
    arclen = sum(seglen);

    % we can quit if the method was 'linear'.
    if strcmpi(method,'linear')
      % we are now done. just exit
      return
    end

    % 'spline' or 'pchip' must have been indicated,
    % so we will be doing an integration. Save the
    % linear chord lengths for later use.
    chordlen = seglen;

    % compute the splines
    spl = cell(1,nd);
    spld = spl;
    diffarray = [3 0 0;0 2 0;0 0 1;0 0 0];
    for i = 1:nd
      switch method
        case 'pchip'
          spl{i} = pchip([0;cumsum(chordlen)],data(:,i));
        case 'spline'
          spl{i} = spline([0;cumsum(chordlen)],data(:,i));
          nc = numel(spl{i}.coefs);
          if nc < 4
            % just pretend it has cubic segments
            spl{i}.coefs = [zeros(1,4-nc),spl{i}.coefs];
            spl{i}.order = 4;
          end
      end

      % and now differentiate them
      xp = spl{i};
      xp.coefs = xp.coefs*diffarray;
      xp.order = 3;
      spld{i} = xp;
    end

    % numerical integration along the curve
    polyarray = zeros(nd,3);
    for i = 1:spl{1}.pieces
      % extract polynomials for the derivatives
      for j = 1:nd
        polyarray(j,:) = spld{j}.coefs(i,:);
      end

      % integrate the arclength for the i'th segment
      % using quadgk for the integral. I could have
      % done this part with an ode solver too.
      seglen(i) = quadgk(@(t) segkernel(t),0,chordlen(i));
    end

    % and sum the segments
    arclen = sum(seglen);

function [path,p_length,p_row,p_col] = shortPath(skeleton,point1,point2,frame_rows,frame_cols)
    %create mask with single point, then get geodesic distance transform
    point1_mask(:,:) = false(frame_rows, frame_cols);
    point1_mask(point1(2),point1(1)) = 1;
    [m,n] = size(point1_mask);
    D1 = bwdistgeodesic(skeleton,point1_mask,'chessboard');

    %create mask with other point, then get geodesic distance transform
    point2_mask(:,:) = false(frame_rows, frame_cols);
    point2_mask(point2(2),point2(1)) = 1;
    D2 = bwdistgeodesic(skeleton,point2_mask,'chessboard');
    
    %find shortest path through skeleton between the points
    D = D1 + D2;
    D(isnan(D)) = inf;
    path = imregionalmin(D);
    
    %find coordinates
    [p_row,p_col] = find(path(:,:)>0);
    %find path length
    p_length = length(p_col(:));
        
    %if the path is the whole frame, there's no path    
    if p_length == frame_rows*frame_cols
        p_row = NaN;
        p_col = NaN;
        p_length = NaN;
        path = NaN;
    end

function [paths,be_closest_tail,be_closest_head,p_length] = traceFish(skeleton,branches_and_ends,branches,head,tail,frame_rows,frame_cols,threshshort,threshlong)

% troubleshooting the function
% clearvars -except skeleton branches_and_ends branches head tail frame_rows frame_cols

%if the closest branch point to the head point is further away than
%this, the closest end point will be selected instead
%thresh = 100;

%get list of branch coordinates
[b_coordinates(:,2),b_coordinates(:,1)] = find(branches(:,:)>0);

%find closest point to head point, starting with branchpoints only
if ~isempty(b_coordinates)==1
    for j=1:length(b_coordinates(:,1))
        head_branch_dist(j) = sqrt((head(1)-b_coordinates(j,1))^2 + (head(2)-b_coordinates(j,2))^2);%%
    end
    [min_dist_head,J] = min(head_branch_dist(:));
end
%include end points if there are no branch points. Check that chosen branch
%point is far enough away to be reasonable.
chk=exist('min_dist_head');
if chk == 1
    if min_dist_head < threshshort %If the initial point choosen is too close
        while min_dist_head < threshshort %If your closest branchpoint is TOO close
            head_branch_dist(J)=Inf; %Set current min to Inf
            prevJ=J;
            [min_dist_head,J] = min(head_branch_dist(:)); %Find next closest point and check again
        end
        if min_dist_head > threshlong %If the closest branchpoint larger than the threshold is too far down the body, use the previous branchpoint.
            J=prevJ;
        end
    elseif min_dist_head > threshlong %If the initial point choosen is too far away, add in endpoints
        clear b_coordinates
        [b_coordinates(:,2),b_coordinates(:,1)] = find(branches_and_ends(:,:)>0);
        for j=1:length(b_coordinates(:,1))
            head_branch_dist(j) = sqrt((head(1)-b_coordinates(j,1))^2 + (head(2)-b_coordinates(j,2))^2);
        end
        [~,J] = min(head_branch_dist(:));
    end

elseif chk==0 %if there are no branchpoints, look at end points
    clear b_coordinates
    [b_coordinates(:,2),b_coordinates(:,1)] = find(branches_and_ends(:,:)>0);
    for j=1:length(b_coordinates(:,1))
        head_branch_dist(j) = sqrt((head(1)-b_coordinates(j,1))^2 + (head(2)-b_coordinates(j,2))^2);
    end
    [~,J] = min(head_branch_dist(:));
end
be_closest_head(1,:) = b_coordinates(J,:);

%now for the tail
p_length = 10001;
loopcount_tail = 1;
while p_length > 10000
    
    %find closest point to tail
    clear be_coordinates tail_branch_dist j
    [be_coordinates(:,2),be_coordinates(:,1)] = find(branches_and_ends(:,:)>0);
    for j=1:length(be_coordinates(:,1))
        tail_branch_dist(j) = sqrt((tail(1)-be_coordinates(j,1))^2 + (tail(2)-be_coordinates(j,2))^2);
    end
    [~,K] = min(tail_branch_dist(:));
    be_closest_tail = be_coordinates(K,:);
    
    %find the shortest path along the skeleton between the two points
    [paths,p_length,~,~] = shortPath(skeleton,be_closest_head(:,1:2),be_closest_tail(:,1:2),frame_rows,frame_cols);
    
    %if the path is extremely long, there's a problem in tail point
    %selection, clear that point and start again
    if p_length > 10000
        branches_and_ends(be_closest_tail(1,1),be_closest_tail(1,2)) = 0;
        clear be_coordinates;
    end
    loopcount_tail = loopcount_tail + 1;
end
    
function val = segkernel(t)
    % sqrt((dx/dt)^2 + (dy/dt)^2)
    
    val = zeros(size(t));
    for k = 1:nd
      val = val + polyval(polyarray(k,:),t).^2;
    end
    val = sqrt(val);

function [paths,be_closest_tail,be_closest_head,p_length] = nearCircle(skeleton,tail,ends,frame_rows,frame_cols,be_closest_tail,be_closest_head,p_thresh)

% troubleshooting the function
% clearvars -except ends skeleton tail frame_rows frame_cols ...
% be_closest_tail b_closest_head 
    
    %find end coordinates (don't bother with branch points) 
    [e(:,2),e(:,1)] = find(ends(:,:)>0);

    %remove original be_closest_tail from the list of end points (if it's 
    %there)
    x = find(e(:,1) == be_closest_tail(:,1));
    if ~isempty(x)
        if isempty(find(e(:,2) == be_closest_tail(:,2)))==1
            e(x(1),:) = Inf;
        elseif isempty(find(e(:,2) == be_closest_tail(:,2)))==0
            y = find(e(:,2) == be_closest_tail(:,2));
        elseif x(1) == y(1)
            e(x(1),:) = Inf;
        end
    end
    %(head point is presumably fine, keep it)

    %calculate distance between all end points and tail point
    for i = 1:length(e(:,1))
        tdist(i,1) = sqrt((tail(1)-e(i,1))^2 + (tail(2)-e(i,2))^2);
    end

    p_length = 1;
    r=1; %loopcount
    
    %while the path is too short to be realistic
    while p_length < p_thresh
        if any(isfinite(tdist))
        %find closest end point to tail
        [~,ind] = min(tdist);
        else %you've gone through all the options...clearly the threshold was set too high, lower slightly and try again.
            fprintf('\nthreshold recalculated: %9.2f\n',p_thresh)
            for i = 1:length(e(:,1))
                tdist(i,1) = sqrt((tail(1)-e(i,1))^2 + (tail(2)-e(i,2))^2);
            end
            p_thresh=p_thresh*0.9;
        end

        %run shortPath using that end point
        [paths,p_length,~,~] = shortPath(skeleton,be_closest_head(:),e(ind,:),frame_rows,frame_cols);

        %if the path is too short, make that tailpoint-endpoint distance
        %Inf, and repeat
        if p_length < p_thresh
            tdist(ind) = Inf;
            clear paths
        end
        r = r+1;
        be_closest_tail = e(ind,:);
    end

function [alt_path] = clickJoin(paths,frame2,head,tail)

% troubleshooting the function
% clearvars -except paths frame2 tail be_closest_tail b_closest_head head
      
    but1 = 1;
    while but1~=32
        clear temp_path man_path tail_join tail_join_mask alt_path pcoord
        but2 = 1;
        z = 0;
        clf;
        
        %find the existing path coordinates and order them head-to-tail
        [pcoord(:,2),pcoord(:,1)] = find(paths(:,:)>0);
        [plhdr] = orderPts(pcoord(:,1:2),head);
        clear pcoord
        pcoord = plhdr;
        clear plhdr 
        
        %plot path over grayscale image
        imshow(frame2); hold on;
        plot(pcoord(:,1),pcoord(:,2),'co');
%         plot(pcoord(1,1),pcoord(1,2),'ro');
        plot(tail(1),tail(2),'ro');
        title('click points to join midline and tail, press space bar when done');
        while but2~=32
            if exist('temp_path','var')==1
                plot(temp_path(z,1),temp_path(z,2),'mo');
            end
            z = z + 1;
            [temp_path(z,1),temp_path(z,2),but2] = ginput(1);
        end

        man_path = temp_path(1:z-1,:);
        
        %find the closest point on the existing path to the first clicked
        %point
        clear dist %I dunno why this is necessary, but it's the only logical explanation for some non-repeatable errors...
        for k = 1:length(pcoord(:,1))
            dist(k) = sqrt((man_path(1,1)-pcoord(k,1))^2 + (man_path(1,2)-pcoord(k,2))^2);
        end
        dist(1:round(size(dist,2)/3))=NaN; %get rid of the first third of dist so it only seaches the back half.
        [~,I] = min(dist); %Search ONLY the back half of dist.
        
        %get rid of all points further towards the tail than that one
        plhdr = pcoord(1:I,:);
        clear pcoord
        pcoord = plhdr;
        clear plhdr 
        paths = zeros(size(frame2,1),size(frame2,2));
        for i = 1:length(pcoord(:,1))
            paths(pcoord(i,2),pcoord(i,1)) = 1;
        end
        paths = logical(paths);
        
        %join clicked points with straight lines, and also join the whole
        %thing to head and tail points
        clf;
        imshow(frame2); hold on;
        plot(pcoord(:,1),pcoord(:,2),'co');
        tail_join(1) = imline(gca,[pcoord(I,1) man_path(1,1)], [pcoord(I,2) man_path(1,2)]);
        head_join = imline(gca,[pcoord(1,1) head(1)], [pcoord(1,2) head(2)]); 
        for g = 2:length(man_path(:,1))
            tail_join(g) = imline(gca,[man_path(g-1,1) man_path(g,1)], [man_path(g-1,2) man_path(g,2)]);
        end
        tail_join(g+1) = imline(gca,[man_path(g,1) tail(1)], [man_path(g,2) tail(2)]);
        alt_path = paths;
        for j = 1:length(tail_join(1,:))
            tail_join_mask(:,:,j) = tail_join(1,j).createMask();
            alt_path = imadd(alt_path, tail_join_mask(:,:,j));
            alt_path = logical(alt_path);
        end
        head_join_mask = head_join.createMask();
        alt_path = imadd(alt_path, head_join_mask);
        alt_path = logical(alt_path);
        
        alt_path = bwmorph(alt_path,'bridge'); %bridge unconnected pixels just in case

        %check with user if path looks ok
        clf;
        imshow(alt_path);
        hold on;
        plot(head(1),head(2),'ro');
        plot(tail(1),tail(2),'ro');
        title('press space bar if satisfactory, otherwise press any other key');
        [~,~,but1] = ginput(1);
    end    
    
function [alt_path] = clickCircle(frame2,head,tail,frame_rows,frame_cols)

but1 = 1;
while but1~=32
    clear temp_path man_path join join_mask alt_path
    but2 = 1;
    z = 0;
    
    %show grayscale image and ask for clicked points
    clf;
    imshow(frame2);
    hold on;
    plot(head(1),head(2),'ro');
    plot(tail(1),tail(2),'bo');
    title('click points to join head to tail, press space bar when done');
    while but2~=32
        if exist('temp_path','var')==1
            plot(temp_path(z,1),temp_path(z,2),'co');
        end
        z = z + 1;
        [temp_path(z,1),temp_path(z,2),but2] = ginput(1);
    end
    
    man_path = temp_path(1:z-1,:);
    
    %join clicked points with straight lines, and also join the whole
    %thing to head and tail points
    join(1) = imline(gca,[head(1) man_path(1,1)], [head(2) man_path(1,2)]);
    for i = 2:length(man_path(:,1))
        join(i) = imline(gca,[man_path(i-1,1) man_path(i,1)], [man_path(i-1,2) man_path(i,2)]);
    end
    join(i+1) = imline(gca,[man_path(i,1) tail(1)], [man_path(i,2) tail(2)]);
    alt_path = false(frame_rows, frame_cols);
    for j = 1:length(join(1,:))
        join_mask(:,:,j) = join(1,j).createMask();
        alt_path = imadd(alt_path, join_mask(:,:,j));
        alt_path = logical(alt_path);
    end
    
    %alt_path = bwmorph(alt_path,'bridge'); %bridge unconnected pixels just in case

    
    %check with user if path looks ok
    clf;
    imshow(alt_path);
    hold on;
    plot(head(1),head(2),'ro');
    plot(tail(1),tail(2),'ro');
    title('press space bar if satisfactory, otherwise press any other key');
    [~,~,but1] = ginput(1);
end    
            
function [inner, fish_mask] = findFish(mask, Nose)

% troubleshooting the function
% clearvars -except mask frame prev_fish bound_min bound_max i start_frame
    
    %Preset variables
    frame_rows = length(mask(:,1));
    frame_cols = length(mask(1,:));
    Nose = round(Nose);
    
    %Find boundaries in the original image
    mask=imdilate(mask,strel('square',4)); %Just in case there's a missing pixel somewhere...and you've clicked exactly on that pixel...literal magic...
    [B,L,N,A] = bwboundaries(mask, 8); %find all boundaries
    FishRegion = L(Nose(2),Nose(1)); %if the nose in actually within a particular boundary

    if FishRegion == 0 %Otherwise find the nearest boundary
        distances = []; regions = [];
        for j = 1:N
            if length(B{j}) > 10
                distances = [distances; min(pdist2([Nose(2),Nose(1)],B{j},'euclidean'))];
                regions = [regions; j];
            end
        end
        [~, closestIndex] = min(distances);
        FishRegion = regions(closestIndex);
    end
    
    %thisBoundary = B{FishRegion};
    
    
    %Now we need to upscale the original image by a factor of 3 to ensure
    %that poly2mask ends up with the same boundary as was pulled from the
    %original image. For details, see:
    %https://blogs.mathworks.com/steve/2014/03/27/comparing-the-geometries-of-bwboundaries-and-poly2mask/
    mask3 = imresize(mask,3,'nearest'); %Upscale image
    B3 = bwboundaries(mask3); %Find boundaries in the upscaled image (same number as in B)
    thisBoundary = B3{FishRegion}; %Pull the appropriate boundary from the upscaled image
    
    %Smooth the fish boundary to avoid extra branches as much as possible.
    smoothboundary=nan(size(thisBoundary));
    smoothboundary(:,1) = smooth(thisBoundary(:,1), size(thisBoundary,1)/20, 'sgolay');
    smoothboundary(:,2) = smooth(thisBoundary(:,2), size(thisBoundary,1)/20, 'sgolay');
    Bx = (smoothboundary(:,2) + 1)/3;
    By = (smoothboundary(:,1) + 1)/3;
    fish_mask = poly2mask(Bx, By, frame_rows, frame_cols);
    
    fish_mask = bwareaopen(fish_mask,30);
    %Rescale coordinates back to the original image size
    %Bx = (thisBoundary(:,2) + 1)/3;
    %By = (thisBoundary(:,1) + 1)/3;
    %fish_mask = poly2mask(Bx, By, frame_rows, frame_cols); %use these coordinates to find the fish_mask

    %fish_mask = poly2mask(thisBoundary(:,2), thisBoundary(:,1), frame_rows, frame_cols);
    
    if (find(A(:,FishRegion) > 0))>0 %check for circular fish, and in these cases, find inner boundary
        [thisBoundary,inner] = innerCircle(thisBoundary,B,FishRegion,A);
    else
        inner = 0;
    end
    
%  function [inner, fish_mask] = findFish(mask,frame,prev_fish,bound_min,bound_max,i,start_frame)   
%    
%     frame_rows = length(mask(:,1));
%     frame_cols = length(mask(1,:));
%     
%     [B,L,N,A] = bwboundaries(mask, 8); %find all boundaries
% 
%         if i==start_frame %for the first frame
%         ffFig = figure();
%         for k = 1:N
%             bound_n = length(B{k});
%             if bound_n > bound_min && bound_n < bound_max %only look at boundaries in reasonable 'fish' size range
%                 figure(ffFig)
%                 imshow(frame, 'InitialMagnification', 60);
%                 hold on;
%                 boundary = B{k};
%                 plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2);
%                 title('press space bar if fish is highlighted, otherwise press any other key');
%                 [~,~,but] = ginput(1);
%                 hold off;
%                 if but == 32 %once fish is selected
%                     thisBoundary(:,:) = B{k};
%                     if (find(A(:,k) > 0))>0 %check for circular fish, and in these cases, find inner boundary 
%                         [thisBoundary,inner] = innerCircle(thisBoundary,B,k,A);
%                     else
%                         inner = 0;
%                     end
%                     %create mask containing only the fish
%                     fish_mask(:,:) = poly2mask(thisBoundary(:,2),thisBoundary(:,1), frame_rows, frame_cols);
%                     break;
%                 end
%            end 
%         end
%         close(ffFig)
% %         fishPixels = ImageStats(k).PixelList
%     else %after first frame, find boundary closest to previous by image subtraction
%         for k = 1:N
%             bound_n = length(B{k});
%             if bound_n > bound_min && bound_n < bound_max
%                 boundary = B{k};
%                 bound_mask(:,:) = poly2mask(boundary(:,2), boundary(:,1), frame_rows, frame_cols);
%                 differences(:,:) = imsubtract(prev_fish,bound_mask(:,:)); 
%                 NumberOfOnes(k) = length(find(differences(:,:)>0));
%                 if NumberOfOnes(k) == 0
%                     NumberOfOnes(k) = Inf;
%                 end
%             else
%                 NumberOfOnes(k) = Inf;
%             end
%         end
%         [~,I] = min(NumberOfOnes(:));
%         thisBoundary = B{I};
%         if (find(A(:,I) > 0))>0
%             [thisBoundary,inner] = innerCircle(thisBoundary,B,I,A); %check for circular fish, and in these cases, find inner boundary
%         else
%             inner = 0;
%         end
%         %create mask containing only the fish
%         fish_mask(:,:) = poly2mask(thisBoundary(:,2), thisBoundary(:,1), frame_rows, frame_cols);
%     end    
    
function [thisBoundary,inner] = innerCircle(thisBoundary,B,k,A)
    b_length = length(thisBoundary(:,1));
    %loop through the children of boundary k
    for l = find(A(:,k))'
        innerBound = B{l};
        i_length = length(innerBound(:,1));
        %find boundary large enough to be fish, if it exists
        if i_length > b_length/4 
            %add to fish boundary
            thisBoundary(b_length+1:i_length+b_length,:) = innerBound; 
            inner = b_length+1;
            fprintf('\n fish is a circle! \n');
            break;
        else
            inner = 0;
        end
    end
    
function [ordered_pts] = orderPts(path_coordinates,head)

    j=1;
    m=1; 
    
    if path_coordinates(1,:)==head
        path_coordinates=path_coordinates(2:end,:);
    end
    
    %find closest path point to the head point
    while j<=length(path_coordinates(:,1)) && isfinite(path_coordinates(j,1)) 
        head_dist(j) = sqrt((head(1)-path_coordinates(j,1))^2 + (head(2)-path_coordinates(j,2))^2);
        j = j+1;
    end
    [~,J] = min(head_dist(:));
    
    %add current point to ordered list, and prevent it from being used
    %again
    current_point(m,:) = path_coordinates(J,:);
    ordered_pts(m,:) = path_coordinates(J,:);
    path_coordinates(J,:) = NaN;
    
    for n = 1:j
        
        %find closest point to current point
        for k = 1:j 
            if k<=length(path_coordinates(:,1)) && isfinite(path_coordinates(k,1))
                point_dist(k) = sqrt((current_point(m,1)-path_coordinates(k,1))^2 + (current_point(m,2)-path_coordinates(k,2))^2);
            else
                point_dist(k) = NaN;
            end
        end
        [~,K] = min(point_dist(:));
        
        %deal with equidistant points, if they exist, by figuring out which
        %point is the most direct route to the rest of the path
        min_rows = find(point_dist(:)==point_dist(K));
        if length(min_rows) > 1
            for q = 1:length(min_rows) 
                for k = 1:j
                    if k ~= min_rows(:)
                        if k<=length(path_coordinates(:,1)) && isfinite(path_coordinates(k,1))
                            next_point_dist(k,q) = sqrt((path_coordinates(min_rows(q),1)-path_coordinates(k,1))^2 + (path_coordinates(min_rows(q),2)-path_coordinates(k,2))^2);
                        else
                            next_point_dist(k,q) = NaN;
                        end
                    else
                        next_point_dist(k,q) = NaN;
                    end
                end
                [dist_compare(q),row_compare(q)] = min(next_point_dist(:,q));
            end
            [~,L] = min(dist_compare);
            K = min_rows(L); %this is for the next point
            min_rows = min_rows(min_rows(:)~=K);
            path_coordinates(min_rows(:),:) = NaN; %set the others to NaN
        end
        
        %add current point to ordered list, and prevent it from being used
        %again
        m = m+1;
        current_point(m,:) = path_coordinates(K,:);
        ordered_pts(m,:) = path_coordinates(K,:);
        path_coordinates(K,:) = NaN;
        clear point_dist next_point_dist min_rows dist_compare;
    end
    %Remove any weird end points that sometimes (apparently randomly)
    %appear at the end of the ordered_pts.
    firstnan=find(isnan(ordered_pts(:,1)),1);
    if pdist2(ordered_pts(1,:),ordered_pts(firstnan-1,:))<pdist2(ordered_pts(firstnan-2,:),ordered_pts(firstnan-1,:))
        ordered_pts(firstnan-1,:)=NaN;
    end
    
    %Get rid of pesky NaNs at end.
    ordered_pts=rmmissing(ordered_pts);
 
    
    
    