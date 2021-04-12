function dX = leakyReluBackward(~, dZ, X, scale)
% Back-propagation using Leaky Rectified Linear on the host

%   Copyright 2016 The MathWorks, Inc.

% Now scale down the negative inputs
negVals = (X < 0);
dX = dZ - negVals .* (1-scale) .* dZ; % Avoid indexing for speed

end
