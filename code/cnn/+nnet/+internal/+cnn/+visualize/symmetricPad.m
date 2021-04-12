function I = symmetricPad(I,padSize)
% Symmetric padding around first two dimensions using 'both' padding style
% from paddarray.
%
% Inputs:
% -------
%   I       : Image to pad. Image maybe a 3 or 4-D array. Only the first
%             two dimensions of the image are padded.
%
%   padSize : padding amount in each dimension.  numel(padSize) must equal 2. 

% Copyright 2016 The MathWorks, Inc.

assert(numel(padSize)==2, 'padSize must contain 2 elements');

idx = iSymmetricPadIndices(size(I), padSize);

I = I(idx{:});


%--------------------------------------------------------------------------
function idx = iSymmetricPadIndices(imgSize, padSize)
numImgDims = numel(imgSize);
if numImgDims > numel(padSize)
    % append 0 to padSize for remaining image dims
    padSize = [padSize repelem(0, numImgDims-2)];
end

numDims = numel(padSize);

% Form index vectors to subsasgn input array into output array.
% Also compute the size of the output array.
idx   = cell(1,numDims);
for k = 1:numDims
    M = imgSize(k);
    
    dimNums = uint32([1:M M:-1:1]);
    
    p = padSize(k);
        
    idx{k}   = dimNums(mod(-p:M+p-1, 2*M) + 1);
    
end