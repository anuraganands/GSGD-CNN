function Z = tanhForward(X)
% tanhForward   Tanh activation
%
% Input:
% X - The input feature maps for a set of images. A (H)x(W)x(C)x(N) array.
%
% Output:
% Z - The output feature maps for a set of images. A (H)x(W)x(C)x(N) array.

%   Copyright 2015-2016 The MathWorks, Inc.

Z = tanh(X);
end
