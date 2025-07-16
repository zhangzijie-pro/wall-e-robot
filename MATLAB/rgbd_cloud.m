videoFile = 'h.mp4';
vReader = VideoReader(videoFile);

focalLength = [407.6, 407.6];
principalPoint = [320, 240];
imageSize = [640, 480];
intrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);

vSet = viewSet;

frame1 = readFrame(vReader);
gray1 = im2gray(frame1);
points1 = detectORBFeatures(gray1);
[features1, validPoints1] = extractFeatures(gray1, points1);

initialPose = rigid3d(eye(4));
vSet = addView(vSet, 1, 'Points', validPoints1, 'Orientation', initialPose.Rotation, 'Location', initialPose.Translation);

prevPoints = validPoints1;
prevFeatures = features1;
prevGray = gray1;
prevPose = initialPose;
frameIdx = 2;

scalePerSecond = 0.05;
frameRate = vReader.FrameRate;
scalePerFrame = scalePerSecond / frameRate;

ptCloudAll = [];  % 保存所有RGB-D点云

while hasFrame(vReader)
    frame = readFrame(vReader);
    gray = im2gray(frame);

    points = detectORBFeatures(gray);
    [features, validPoints] = extractFeatures(gray, points);

    indexPairs = matchFeatures(prevFeatures, features, 'Unique', true);
    matchedPoints1 = prevPoints(indexPairs(:,1));
    matchedPoints2 = validPoints(indexPairs(:,2));

    if matchedPoints1.Count < 8
        continue;
    end

    [E, inlierIdx] = estimateEssentialMatrix(matchedPoints1, matchedPoints2, intrinsics, 'Confidence', 99.99);
    inlierPoints1 = matchedPoints1(inlierIdx);
    inlierPoints2 = matchedPoints2(inlierIdx);

    [orient, loc] = relativeCameraPose(E, intrinsics, inlierPoints1, inlierPoints2);

    locNorm = norm(loc);
    scaledLoc = loc * (scalePerFrame / max(locNorm, eps));

    relPose = rigid3d(orient, scaledLoc);
    absPose = rigid3d(relPose.T * prevPose.T);

    vSet = addView(vSet, frameIdx, 'Points', validPoints, 'Orientation', absPose.Rotation, 'Location', absPose.Translation);
    vSet = addConnection(vSet, frameIdx-1, frameIdx, 'Matches', indexPairs);

    % Triangulate for depth
    camMatrix1 = cameraMatrix(intrinsics, prevPose.Rotation, prevPose.Translation);
    camMatrix2 = cameraMatrix(intrinsics, absPose.Rotation, absPose.Translation);
    worldPoints = triangulate(matchedPoints1, matchedPoints2, camMatrix1, camMatrix2);

    % Get depth and colors
    depths = worldPoints(:,3);
    colors = zeros(size(worldPoints,1), 3);
    for i = 1:size(matchedPoints2,1)
        x = round(matchedPoints2.Location(i,1));
        y = round(matchedPoints2.Location(i,2));
        if x > 0 && x <= imageSize(2) && y > 0 && y <= imageSize(1)
            colors(i, :) = double(reshape(frame(y, x, :), 1, 3))/255;
        end
    end

    validIdx = depths > 0 & all(colors > 0, 2);
    ptCloudFrame = pointCloud(worldPoints(validIdx,:), 'Color', colors(validIdx,:));

    if ptCloudFrame.Count > 0
        if isempty(ptCloudAll)
            ptCloudAll = ptCloudFrame;
        else
            ptCloudAll = pcmerge(ptCloudAll, ptCloudFrame, 0.05);
        end
    end


    prevPoints = validPoints;
    prevFeatures = features;
    prevGray = gray;
    prevPose = absPose;

    frameIdx = frameIdx + 1;
end

figure('Name', 'Merged RGB-D Point Cloud');
pcshow(ptCloudAll);
title('Accumulated RGB-D Point Cloud');

pcwrite(ptCloudAll, 'rgbd_pointcloud.ply');
