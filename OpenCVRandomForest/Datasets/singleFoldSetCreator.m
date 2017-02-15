function [ trainIdx, testIdx] = singleFoldSetCreator( AllIdx, singleFoldPercent )
%SINGLEFOLDSETCREATOR Returns indexes of singlefold training and testing
%data
numSamples = size(AllIdx, 2);
numTesting = floor(singleFoldPercent*numSamples);

dataDivide = false(1, numSamples);
[~, idx] = datasample(dataDivide, numTesting, 2, 'Replace', false);
dataDivide(idx) = 1;

trainIdx = zeros(size(AllIdx,1), numSamples-numTesting);
testIdx = zeros(size(AllIdx,1), numTesting);

% dividing the data into training and testing sets
trIdx = 1;
tsIdx = 1;
for i = 1: numSamples
    
    if(dataDivide(1, i) == 0)
       trainIdx(:, trIdx) = AllIdx(:,i);
        trIdx = trIdx +1 ;
    else
        testIdx(:, tsIdx) = AllIdx(:,i);
        tsIdx = tsIdx+1;
        
    end
end

end

