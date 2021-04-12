function X = halton(outputSize, skip, leap)
% halton   Return a tensor X of size outputSize which elements are
% points from the Halton sequence
%
% Inputs:
%   outputSize    - Size of the output tensor X
%   skip          - Number of initial points to omit from sequence
%   leap          - Interval between points
%

%   Copyright 2017 The MathWorks, Inc.

n = prod(outputSize);
X = zeros(n,1);
Idx = 1;
for haltonIdx = 1+skip:leap:skip+n*leap
    X(Idx) = singlePointHalton(haltonIdx,2);
    Idx = Idx+1;
end
% Make sure outputSize has at least two components to be a valid size
outputSize = [outputSize 1];
% Reshape X and shift between [-1 1]
X = reshape(X, outputSize)*2 - 1;
end

function x = singlePointHalton(idx, b)
% singlePointHalton   Get a single point from the Halton sequence
% corresponding to the index idx using base b
x = 0;
radix = b;
while idx>0
    % Divide by base and work out remainder
    idxNew = floor(idx/b);
    a = idx - b.*idxNew;
    idx = idxNew;
    
    x = x + a./radix;
    radix = radix*b;
end
end