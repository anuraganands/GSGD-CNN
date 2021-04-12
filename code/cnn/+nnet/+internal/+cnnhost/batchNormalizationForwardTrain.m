function [Z,batchMean,batchInvVar] = batchNormalizationForwardTrain(X, beta, gamma, epsilon)
% Forward batch normalization on the host, training phase
% Returns the layer output and the batch mean and inverse variance
    
%   Copyright 2016-2017 The MathWorks, Inc.

m = numel(X) ./ size(X,3); % total number of elements in batch per activation
batchMean = iSumAllExcept3D(X) ./ m;
batchVar = iSumAllExcept3D( (X - batchMean).^2 ) ./ m; 

batchInvVar = 1./sqrt(batchVar + epsilon);

scale = gamma .* batchInvVar;
offset = beta - batchMean.*scale;

Z = scale.*X + offset;

end

function out = iSumAllExcept3D(in)
% Helper to sum a 4D array in all dimensions except the third:
%  (HxWxCxN) -> (1x1xCx1)
    [d1, d2, d3, d4] = size(in);
    out = reshape( sum( sum( reshape(in, d1*d2, d3, d4), 1 ), 3 ), [1, 1, d3, 1] );
end