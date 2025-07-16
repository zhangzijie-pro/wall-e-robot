videoFile = 'h.mp4';
vReader = VideoReader(videoFile);

focalLength = [407.6, 407.6];  % fx, fy
principalPoint = [320, 240];  % cx, cy

imageSize = [640, 480];
intrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);

% 初始化视图集合
vSet = viewSet;

frame1 = readFrame(vReader);
gray1 = im2gray(frame1);
points1 = detectORBFeatures(gray1);
[features1, validPoints1] = extractFeatures(gray1, points1);

initialPose = rigid3d(eye(4));  % 单位变换矩阵
vSet = addView(vSet, 1, 'Points', validPoints1, 'Orientation', initialPose.Rotation, 'Location', initialPose.Translation);
fprintf('Added view 1 with %d points\n', validPoints1.Count);

prevPoints = validPoints1;
prevFeatures = features1;
prevGray = gray1;
prevPose = initialPose;
frameIdx = 2;

while hasFrame(vReader)
    frame = readFrame(vReader);
    gray = im2gray(frame);

    points = detectORBFeatures(gray);
    [features, validPoints] = extractFeatures(gray, points);

    indexPairs = matchFeatures(prevFeatures, features, 'Unique', true);
    matchedPoints1 = prevPoints(indexPairs(:,1));
    matchedPoints2 = validPoints(indexPairs(:,2));

    if matchedPoints1.Count < 8
        fprintf('Not enough matches in frame %d. Skipping.\n', frameIdx);
        continue;
    end

    [E, inlierIdx] = estimateEssentialMatrix(matchedPoints1, matchedPoints2, intrinsics, 'Confidence', 99.99);
    inlierPoints1 = matchedPoints1(inlierIdx);
    inlierPoints2 = matchedPoints2(inlierIdx);

    [orient, loc] = relativeCameraPose(E, intrinsics, inlierPoints1, inlierPoints2);
    relPose = rigid3d(orient, loc);
    absPose = rigid3d(relPose.T * prevPose.T);  % 累积绝对姿态

    vSet = addView(vSet, frameIdx, 'Points', validPoints, 'Orientation', absPose.Rotation, 'Location', absPose.Translation);
    fprintf('Added view %d with %d points\n', frameIdx, validPoints.Count);

    % 修正：先添加连接再更新相对姿态
    vSet = addConnection(vSet, frameIdx-1, frameIdx, 'Matches', indexPairs);
    vSet = updateConnection(vSet, frameIdx-1, frameIdx, 'Orientation', relPose.Rotation, 'Location', relPose.Translation);
    fprintf('Added connection between frames %d and %d\n', frameIdx-1, frameIdx);

    prevPoints = validPoints;
    prevFeatures = features;
    prevGray = gray;
    prevPose = absPose;

    frameIdx = frameIdx + 1;
end

% 稀疏点云重建
tracks = findTracks(vSet);
cameraPoses = poses(vSet);
xyzPoints = triangulateMultiview(tracks, cameraPoses, intrinsics);

% 可视化稀疏点云
figure;
pcshow(xyzPoints, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', 'MarkerSize', 45);
hold on;
% Plot first camera (ViewId 1, typically at origin with identity rotation)
plotCamera('Location', cameraPoses.Location{1}, 'Orientation', cameraPoses.Orientation{1}, 'Size', 0.2, 'Color', 'r', 'Label', '1', 'Opacity', 0.5);
% Plot subsequent cameras
for i = 2:height(cameraPoses)
    plotCamera('Location', cameraPoses.Location{i}, 'Orientation', cameraPoses.Orientation{i}, 'Size', 0.2, 'Color', 'b', 'Label', num2str(i), 'Opacity', 0.5);
end
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Sparse Point Cloud with Camera Trajectory');
grid on;

% 导出为 PLY 文件
ptCloud = pointCloud(xyzPoints);
pcwrite(ptCloud, 'output_sparse_cloud.ply');