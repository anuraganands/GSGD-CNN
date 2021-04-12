function Z = leakyReluForward(X, scale)
% Forward activation using Leaky Rectified Linear on the host
    
%   Copyright 2016 The MathWorks, Inc.

Z = max(0,X,'includenan') + scale.*min(0,X);

end
