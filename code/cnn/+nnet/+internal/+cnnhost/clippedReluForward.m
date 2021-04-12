function Z = clippedReluForward(X, ceiling)
% Forward activation using Clipped Rectified Linear on the host
    
%   Copyright 2016 The MathWorks, Inc.

Z = min(max(0, X), ceiling);

end
