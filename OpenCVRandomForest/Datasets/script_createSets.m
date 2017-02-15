% script for creating training and test datasets for data
close all
clear all
clc

singleFoldPercent = 0.3
% load the 3D Road Network data
inFeatures = ReadMatFromFile('3D_RoadNetwork\\Converted\\3D_Spatial_InputFeatures.dat');
inTarget = ReadMatFromFile('3D_RoadNetwork\\Converted\\3D_Spatial_TargetVariable.dat');

inFeatures = inFeatures';
inTarget = inTarget';

inIdx = 1:size(inFeatures, 2);

[ trainIdx, testIdx] = singleFoldSetCreator( inIdx,  singleFoldPercent);

trainFeatures = zeros(size(inFeatures, 1), size(trainIdx, 2));
trainTarget = zeros(size(inTarget, 1), size(trainIdx, 2));

for i = 1:size(trainIdx, 2)
    trainFeatures(:, i) = inFeatures(:, trainIdx(i));
    trainTarget(:, i) = inTarget(:, trainIdx(i));
end

testFeatures = zeros(size(inFeatures, 1), size(testIdx, 2));
testTarget = zeros(size(inTarget, 1), size(testIdx, 2));

for i = 1:size(testIdx, 2)
    testFeatures(:, i) = inFeatures(:, testIdx(i));
    testTarget(:, i) = inTarget(:, testIdx(i));
end

WriteMatToFile(trainFeatures, '3D_RoadNetwork\\Singlefold\\RoadNetwork_TrainFeatures.dat');
WriteMatToFile(testFeatures, '3D_RoadNetwork\\Singlefold\\RoadNetwork_TestFeatures.dat');

WriteMatToFile(trainTarget, '3D_RoadNetwork\\Singlefold\\RoadNetwork_TrainTarget.dat');
WriteMatToFile(testTarget, '3D_RoadNetwork\\Singlefold\\RoadNetwork_TestTarget.dat');


% load the AirFoil Data
inFeatures = ReadMatFromFile('AirFoil\\Converted\\AirFoil_InputFeatures.dat');
inTarget = ReadMatFromFile('AirFoil\\Converted\\AirFoil_TargetVariable.dat');

inFeatures = inFeatures';
inTarget = inTarget';

inIdx = 1:size(inFeatures, 2);

[ trainIdx, testIdx] = singleFoldSetCreator( inIdx,  singleFoldPercent);

trainFeatures = zeros(size(inFeatures, 1), size(trainIdx, 2));
trainTarget = zeros(size(inTarget, 1), size(trainIdx, 2));

for i = 1:size(trainIdx, 2)
    trainFeatures(:, i) = inFeatures(:, trainIdx(i));
    trainTarget(:, i) = inTarget(:, trainIdx(i));
end

testFeatures = zeros(size(inFeatures, 1), size(testIdx, 2));
testTarget = zeros(size(inTarget, 1), size(testIdx, 2));

for i = 1:size(testIdx, 2)
    testFeatures(:, i) = inFeatures(:, testIdx(i));
    testTarget(:, i) = inTarget(:, testIdx(i));
end

WriteMatToFile(trainFeatures, 'AirFoil\\Singlefold\\AirFoil_TrainFeatures.dat');
WriteMatToFile(testFeatures, 'AirFoil\\Singlefold\\AirFoil_TestFeatures.dat');

WriteMatToFile(trainTarget, 'AirFoil\\Singlefold\\AirFoil_TrainTarget.dat');
WriteMatToFile(testTarget, 'AirFoil\\Singlefold\\AirFoil_TestTarget.dat');



% load the Breast Cancer data
inFeatures = ReadMatFromFile('BreastCancer\\Converted\\BreastCancer_InputFeatures.dat');
inTargetRegression = ReadMatFromFile('BreastCancer\\Converted\\BreastCancer_TargetVariableRegression.dat');
inTargetClassification = ReadMatFromFile('BreastCancer\\Converted\\BreastCancer_TargetVariableClassification.dat');

inFeatures = inFeatures';
inTargetRegression = inTargetRegression';
inTargetClassification = inTargetClassification';

inIdx = 1:size(inFeatures, 2);

[ trainIdx, testIdx] = singleFoldSetCreator( inIdx,  singleFoldPercent);

trainFeatures = zeros(size(inFeatures, 1), size(trainIdx, 2));
trainTargetRegression = zeros(size(inTargetRegression, 1), size(trainIdx, 2));
trainTargetClassification = zeros(size(inTargetClassification, 1), size(trainIdx, 2));

for i = 1:size(trainIdx, 2)
    trainFeatures(:, i) = inFeatures(:, trainIdx(i));
    trainTargetRegression(:, i) = inTargetRegression(:, trainIdx(i));
    trainTargetClassification(:, i) = inTargetClassification(:, trainIdx(i));
end

testFeatures = zeros(size(inFeatures, 1), size(testIdx, 2));
testTargetRegression = zeros(size(inTargetRegression, 1), size(testIdx, 2));
testTargetClassification = zeros(size(inTargetClassification, 1), size(testIdx, 2));

for i = 1:size(testIdx, 2)
    testFeatures(:, i) = inFeatures(:, testIdx(i));
    testTargetRegression(:, i) = inTargetRegression(:, testIdx(i));
    testTargetClassification(:, i) = inTargetClassification(:, testIdx(i));
end

WriteMatToFile(trainFeatures, 'BreastCancer\\Singlefold\\BreastCancer_TrainFeatures.dat');
WriteMatToFile(testFeatures, 'BreastCancer\\Singlefold\\BreastCancer_TestFeatures.dat');

WriteMatToFile(trainTargetRegression, 'BreastCancer\\Singlefold\\BreastCancer_TrainTargetRegression.dat');
WriteMatToFile(testTargetRegression, 'BreastCancer\\Singlefold\\BreastCancer_TestTargetRegression.dat');

WriteMatToFile(trainTargetClassification, 'BreastCancer\\Singlefold\\BreastCancer_TrainTargetClassification.dat');
WriteMatToFile(testTargetClassification, 'BreastCancer\\Singlefold\\BreastCancer_TestTargetClassification.dat');