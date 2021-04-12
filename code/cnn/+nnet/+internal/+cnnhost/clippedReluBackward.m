function dX = clippedReluBackward(~, dZ, X, ceiling)
% Back-propagation using Clipped Rectified Linear on the host

%   Copyright 2016 The MathWorks, Inc.

% Gradient is zero for non-positive or above-ceiling values
dX = dZ;
dX(X<=0 | X>ceiling) = 0;

end
