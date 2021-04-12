function dLossdX = localMapNormBackward2D(Z, dLossdZ, X, windowSize, alpha, beta, k) %#ok<INUSL>
% localMapNormBackward2D   Perform backpropagation for local map normalization
%
% Inputs:
% Z - The output from the local map normalization layer. Not used in the
% current implementation but present to match cuDNN.
% dLossdZ - The derivative of the loss function with respect to the output
% of the normalization layer. A (H)x(W)x(C)x(N) array.
% X - The input feature maps for a set of images. A (H)x(W)x(C)x(N) array.
% windowSize - The number of maps to use for the normalization of each
% element.
% alpha - Multiplier for the normalization term.
% beta - Exponent for the normalization term.
% k - Offset for the normalization term.
%
% Output:
% dLossdZ - The derivative of the loss function with respect to the input
% of the normalization layer. A (H)x(W)x(C)x(N) array.

%   Copyright 2015-2017 The MathWorks, Inc.

dLossdX = builtin('_xchannelnorm', X, windowSize, alpha, beta, k, dLossdZ);
