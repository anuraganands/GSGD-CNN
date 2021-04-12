function Z = softmaxForward2D(X)
% softmaxForward2D   Forward propagation for softmax on the host
%   Z = softmaxForward2D(X) computes the softmax activation on X, and
%   returns it as Z.
%
%   Input:
%   X - The input vectors for a set of images. A (H)x(W)x(D)x(N) array.
%
%   Output:
%   Z - The output vectors for a set of images. A (H)x(W)x(D)x(N) array.

%   Copyright 2016 The MathWorks, Inc.

exponents = X - max(X,[],3);
expX = exp(exponents);
Z = expX./sum(expX,3);
end