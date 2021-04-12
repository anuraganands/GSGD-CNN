function [dX,dBeta,dGamma] = batchNormalizationBackward(dZ, X, gamma, epsilon, batchMean, batchInvVar) %#ok<INUSL>
% Back-propagation using batch normalization layer on the host
% NB: batchInvVar is actually 1./sqrt(var(X) + epsilon)

%   Copyright 2016-2017 The MathWorks, Inc.

Xnorm = (X - batchMean) .* batchInvVar;

% Get the gradient of the function w.r.t the parameters beta and gamma.
dBeta = iSumAllExcept3D(dZ);
dGamma = iSumAllExcept3D(dZ .* Xnorm);

% Now get the gradient of the function w.r.t. input (x)
% See Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network
% Training by Reducing Internal Covariate Shift" for details.

m = numel(X) ./ size(X,3); % total number of elements in batch per activation
factor = gamma .* batchInvVar;
factorScaled = factor ./ m;

dMean = dBeta .* factorScaled;
dVar = dGamma .* factorScaled;

dX = dZ .* factor - Xnorm .* dVar - dMean;

end


function out = iSumAllExcept3D(in)
% Helper to sum a 4D array in all dimensions except the third:
%  (HxWxCxN) -> (1x1xCx1)
    [d1, d2, d3, d4] = size(in);
    out = reshape( sum( sum( reshape(in, d1*d2, d3, d4), 1 ), 3 ), [1, 1, d3, 1] );
end