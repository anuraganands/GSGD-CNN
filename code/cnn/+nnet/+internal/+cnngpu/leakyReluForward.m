function Z = leakyReluForward(X, scale)
% Forward activation using Leaky Rectified Linear on the GPU
    
%   Copyright 2016 The MathWorks, Inc.

% Ensure calculation on GPU even for host-side inputs
X = gpuArray(X); % No-op if already on GPU

Z = max(0,X,'includenan') + scale.*min(0,X);

end
