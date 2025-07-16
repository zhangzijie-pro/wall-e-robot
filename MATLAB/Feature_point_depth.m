videoFile = 'h.mp4';
vReader = VideoReader(videoFile);

focalLength = [426.67, 426.67];
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

depthVideo = VideoWriter('depth_output.mp4', 'MPEG-4');
open(depthVideo);

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
    scaledLoc = loc * (scalePerFrame / max(locNorm, eps));  % 避免除零

    relPose = rigid3d(orient, scaledLoc);
    absPose = rigid3d(relPose.T * prevPose.T);

    vSet = addView(vSet, frameIdx, 'Points', validPoints, 'Orientation', absPose.Rotation, 'Location', absPose.Translation);
    vSet = addConnection(vSet, frameIdx-1, frameIdx, 'Matches', indexPairs);

    % Triangulate matched points to create depth map
    camMatrix1 = cameraMatrix(intrinsics, prevPose.Rotation, prevPose.Translation);
    camMatrix2 = cameraMatrix(intrinsics, absPose.Rotation, absPose.Translation);
    worldPoints = triangulate(matchedPoints1, matchedPoints2, camMatrix1, camMatrix2);

    depths = worldPoints(:,3);
    depthMap = zeros(imageSize);
    for i = 1:size(matchedPoints2, 1)
        x = round(matchedPoints2.Location(i,1));
        y = round(matchedPoints2.Location(i,2));
        if x > 0 && x <= imageSize(2) && y > 0 && y <= imageSize(1)
            depthMap(y, x) = depths(i);
        end
    end

    depthVis = mat2gray(depthMap);
    depthColor = ind2rgb(uint8(depthVis * 255), jet(256));
    writeVideo(depthVideo, im2uint8(depthColor));

    prevPoints = validPoints;
    prevFeatures = features;
    prevGray = gray;
    prevPose = absPose;

    frameIdx = frameIdx + 1;
end

close(depthVideo);
disp('深度视频已保存为 depth_output.mp4');
