function Z = reluForward(X)
% reluForward   Rectified Linear Unit (ReLU) activation on the host
%   Z = reluForward(X) takes the input X and applies the ReLU function to
%   return Z.
%
%   Input:
%   X - Input channels for a set of images. A (H)x(W)x(C)x(N) array.
%
%   Output:
%   Z - Output channels for a set of images. A (H)x(W)x(C)x(N) array.

%   Copyright 2016-2017 The MathWorks, Inc.

Z = max(0,X);
end