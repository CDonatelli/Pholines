function dataStruct = binarizeFish(binaryVideo, videoData)
    
    dataStruct = struct;
    [dataStruct.videoName, dataStruct.videoPath] = uigetfile('*.*', ...
        'Select the video you want to binarize');
    
    fishVideo = VideoReader([dataStruct.videoPath,dataStruct.videoName]);
    FrNum = fishVideo.NumberOfFrames;
        dataStruct.originalRes = [fishVideo.Height, fishVideo.Width];
        dataStruct.format = fishVideo.VideoFormat;
    
    % Video Data
    prompt = {'Fish ID:', 'Trial ID:', 'Frame Rate:', 'Fish Length (mm):','Fish Weight (g):'};
    dlgtitle = 'Fish Data: Leave defaults if unknown';
    dims = [1 35];
    definput = {'n/a','n/a', num2str(fishVideo.FrameRate),'0','0'};
    answer = inputdlg(prompt,dlgtitle,dims,definput);
        dataStruct.fishID = answer{1};
        dataStruct.trialID = answer{2};
        dataStruct.originalFrameRate = str2num(answer{3});
        dataStruct.fishLength = str2num(answer{4});
        dataStruct.fishWeight = str2num(answer{5});
        
        startFrame = 1;
        endFrame = FrNum;
        skipRate = 1;
    
    RawImage = read(fishVideo,startFrame);
    imshow(RawImage)
    
    answer = questdlg('Do you want to crop your video', ...
    'Crop Video', ...
    'Yes','No','No');
    switch answer
        case 'Yes'
            rect = CropVideo(RawImage);
            dataStruct.cropRect = rect;
        case 'No'
            dataStruct.cropRect = 0;
    end
       
    dataStruct.binaryFrameRate = dataStruct.originalFrameRate/skipRate;
    
    choice = 1;
    if dataStruct.cropRect == 0
        if strcmp(convertCharsToStrings(dataStruct.format), 'Grayscale')
            [BackLev, FishLev] = GetLevelsBW(RawImage);
        else
            [BackLev, FishLev] = GetLevels(RawImage);
        end
    else
        RawImage = imcrop(RawImage, rect);
        if strcmp(convertCharsToStrings(dataStruct.format), 'Grayscale')
            [BackLev, FishLev] = GetLevelsBW(RawImage);
        else
            [BackLev, FishLev] = GetLevels(RawImage);
        end
    end
    
    ThreshLevel = median([BackLev(2),FishLev(1)])/255;
    sOut.backLevel = BackLev; sOut.fishLevel = FishLev;
    BinaryImage = ProcessImage(RawImage,ThreshLevel);
    imshow(BinaryImage)
    
    while choice ~= 2
        choice = input('Is this good? (1 = brighten fish, 0 = dim  fish, 2 = good): ');
        if choice == 0
            ThreshLevel = ThreshLevel - 0.025;
            BinaryImage = ProcessImage(RawImage,ThreshLevel);
            imshow(BinaryImage)
        elseif choice == 1
            ThreshLevel = ThreshLevel + 0.025;
            BinaryImage = ProcessImage(RawImage,ThreshLevel);
            imshow(BinaryImage)
        else
            ThreshLevel = ThreshLevel;
        end
    end
    
    bwVideo = VideoWriter([dataStruct.videoPath,binaryVideo],'Grayscale AVI');
    open(bwVideo)
    
    for i = startFrame:skipRate:endFrame        
        %Read in frames
        if dataStruct.cropRect == 0
            RawImage = read(fishVideo,i);
        else
            RawImage = read(fishVideo,i);
            RawImage = imcrop(RawImage, rect);
        end
        
        %Process frame
        [BinaryImage] = ProcessImage(RawImage, ThreshLevel);
        
        writeVideo(bwVideo, double(BinaryImage))
        disp(['Processed frame ',num2str(i),'/',num2str(endFrame)])
    end
    
    close(bwVideo)
    
    bwVideo = VideoReader([dataStruct.videoPath,binaryVideo]);
        dataStruct.croppedRes = [bwVideo.Height, bwVideo.Width];
        dataStruct.bwFormat = bwVideo.VideoFormat;
    
    save([dataStruct.videoPath,videoData], 'dataStruct')


function [FrameOut] = ProcessImage(Frame, Level)
    h = ones(5,5) / 25;
    BlurredImage = imfilter(Frame,h);

    FrameOut = ~im2bw(BlurredImage,Level);
    [m,n] = size(FrameOut);
    FrameOut = bwareaopen(FrameOut, round(0.001*(m*n)));

%remove weird edges from video frame
    FrameOut(1:3,:) = []; FrameOut(end-2:end,:) = []; FrameOut(:,1:3) = []; FrameOut(:,end-2:end) = [];

%Smooth broken bits of fish
FrameOut = bwareaopen(FrameOut, round(0.001*(m*n)));
    se = strel('disk',10);
    FrameOut = imclose(FrameOut,se);

function rect = CropVideo(im)
    disp('Select the portion of the frame the fish swims through');
    choice = 0;
    while choice == 0
        imshow(im)
        rect = getrect;
        im2 = imcrop(im,rect);
        imshow(im2)
        choice = input('Does this look right? :');
    end
    
function [Back, Obj] = GetLevels(im)
    
    % Read in original RGB image.
    rgbImage = im;
    % Extract color channels.
    redChannel = rgbImage(:,:,1); % Red channel
    greenChannel = rgbImage(:,:,2); % Green channel
    blueChannel = rgbImage(:,:,3); % Blue channel
    % Create an all black channel.
    allBlack = zeros(size(rgbImage, 1), size(rgbImage, 2), 'uint8');
    % Create color versions of the individual color channels.
    just_red = cat(3, redChannel, allBlack, allBlack);
    just_green = cat(3, allBlack, greenChannel, allBlack);
    just_blue = cat(3, allBlack, allBlack, blueChannel);
    % Recombine the individual color channels to create the original RGB image again.
    recombinedRGBImage = cat(3, redChannel, greenChannel, blueChannel);
    % Display them all.
    subplot(3, 3, 2);
    imshow(rgbImage);
    fontSize = 20; title('Original RGB Image', 'FontSize', fontSize)
    subplot(3, 3, 4);
    imshow(just_red); title('Red Channel in Red', 'FontSize', fontSize)
    subplot(3, 3, 5);
    imshow(just_green); title('Green Channel in Green', 'FontSize', fontSize)
    subplot(3, 3, 6);
    imshow(just_blue); title('Blue Channel in Blue', 'FontSize', fontSize)
    subplot(3, 3, 8);
    imshow(recombinedRGBImage); title('Recombined to Form Original RGB Image Again', 'FontSize', fontSize)
    
    answer = questdlg('Which channel had the most contrast?', ...
        'Color Channels', ...
        'Red','Green','Blue','Blue');
    switch answer
        case 'Red'
            channel = 1;
        case 'Green'
            channel = 2;
        case 'Blue'
            channel = 3;
    end

    close all
    
    OBlu = []; BBlu = [];
    imshow(im); hold on
    disp('Get Fish Levels');
    [Xo Yo] = getpts;
    plot(Xo,Yo,'bo');
    hold on
    for i = 1:length(Xo)
        O = impixel(im,Xo(i),Yo(i));
        OBlu = [OBlu,O(channel)];
    end
    disp('Get background Levels');
    [Xb Yb] = getpts;
    plot(Xb,Yb,'ro');
    hold on
    for i = 1:length(Xb)
        B = impixel(im,Xb(i),Yb(i));
        BBlu = [BBlu,B(channel)];
    end
    
    MaxObj = max(OBlu); MinObj = min(OBlu);
    MaxBac = max(BBlu); MinBac = min(BBlu);
    
    % If the levels are overlapping, find the average
    % For now assuming that the background and fish are pretty different
    % so the only overlapping levels considered are as follows:
    % MaxFish > MaxBackground > MinFish > MinBackground
    % MaxBackground > MaxFish > MinBackground > MinFish
    % Looking to create an order that looks like one of the following:
    % MaxFish > MinFish > MaxBackground > MinBackground
    % MaxBackground > MinBackground > MaxFish > MinFish
    
    if MaxObj >= MinBac 
        if MaxBac >= MinObj 
            Avg = round(mean([MaxBac MinObj]));
            MinObj = Avg;
            MaxBac = Avg-1;
        end
    end
    if MaxBac >= MinObj 
        if MaxObj >= MinBac   
            Avg = round(mean([MaxObj MinBac]));
            MaxObj = Avg;
            MinBac = Avg+1;
        end
    end
    
    %Now I need to account for the fact that the user might not select the
    %full range of points on the fish (I did this and it leads to incorrect
    %D values which means incorrect wobble measurements. This is a cheat-y
    %fix but it works with my videos. This part of the code is less
    %adaptable for other users (especially if they are filming in a
    %background that is not super different from the fish). 
    %Assumes one of these two cases:
    % MinBackground < MaxBackground < MinFish < MaxFish 
    % MinFish < MaxFish < MinBackground < MaxBackground
    
    if MinObj > MaxBac
        MinObj = MinObj - ((MinObj-MaxBac)/2);
        MaxObj = MaxObj + ((MinObj-MaxBac)/2);
    end
    if MinBac > MaxObj
        MaxObj = MaxObj + ((MinBac-MaxObj)/2);
        MinObj = MinObj - ((MinBac-MaxObj)/2);
    end
    
    hold off
    Back = [MaxBac, MinBac]; Obj = [MaxObj, MinObj];
    
function [Back, Obj] = GetLevelsBW(im)   
    % Read in original RGB image.
    rgbImage = im;
        
    OBlu = []; BBlu = [];
    imshow(im); hold on
    disp('Get Fish Levels');
    [Xo Yo] = getpts;
    plot(Xo,Yo,'bo');
    hold on
    for i = 1:length(Xo)
        O = impixel(im,Xo(i),Yo(i));
        OBlu = [OBlu,O(1)];
    end
    disp('Get background Levels');
    [Xb Yb] = getpts;
    plot(Xb,Yb,'ro');
    hold on
    for i = 1:length(Xb)
        B = impixel(im,Xb(i),Yb(i));
        BBlu = [BBlu,B(1)];
    end
    
    MaxObj = max(OBlu); MinObj = min(OBlu);
    MaxBac = max(BBlu); MinBac = min(BBlu);
    
    % If the levels are overlapping, find the average
    % For now assuming that the background and fish are pretty different
    % so the only overlapping levels considered are as follows:
    % MaxFish > MaxBackground > MinFish > MinBackground
    % MaxBackground > MaxFish > MinBackground > MinFish
    % Looking to create an order that looks like one of the following:
    % MaxFish > MinFish > MaxBackground > MinBackground
    % MaxBackground > MinBackground > MaxFish > MinFish
    
    if MaxObj >= MinBac 
        if MaxBac >= MinObj 
            Avg = round(mean([MaxBac MinObj]));
            MinObj = Avg;
            MaxBac = Avg-1;
        end
    end
    if MaxBac >= MinObj 
        if MaxObj >= MinBac   
            Avg = round(mean([MaxObj MinBac]));
            MaxObj = Avg;
            MinBac = Avg+1;
        end
    end
    
    %Now I need to account for the fact that the user might not select the
    %full range of points on the fish (I did this and it leads to incorrect
    %D values which means incorrect wobble measurements. This is a cheat-y
    %fix but it works with my videos. This part of the code is less
    %adaptable for other users (especially if they are filming in a
    %background that is not super different from the fish). 
    %Assumes one of these two cases:
    % MinBackground < MaxBackground < MinFish < MaxFish 
    % MinFish < MaxFish < MinBackground < MaxBackground
    
    if MinObj > MaxBac
        MinObj = MinObj - ((MinObj-MaxBac)/2);
        MaxObj = MaxObj + ((MinObj-MaxBac)/2);
    end
    if MinBac > MaxObj
        MaxObj = MaxObj + ((MinBac-MaxObj)/2);
        MinObj = MinObj - ((MinBac-MaxObj)/2);
    end
    
    hold off
    Back = [MaxBac, MinBac]; Obj = [MaxObj, MinObj];