function dX = leakyReluBackward(~, dZ, X, scale)
% Back-propagation using Leaky Rectified Linear on the GPU

%   Copyright 2016 The MathWorks, Inc.

% Ensure calculation on GPU even for host-side inputs
X = gpuArray(X); % No-op if already on GPU
dZ = gpuArray(dZ); % No-op if already on GPU

% Now scale down the negative inputs
negVals = (X < 0);
dX = dZ - negVals .* (1-scale) .* dZ; % Avoid indexing for speed

end
