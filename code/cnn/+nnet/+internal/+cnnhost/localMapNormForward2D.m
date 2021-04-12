function Z = localMapNormForward2D(X, windowSize, alpha, beta, k)
% localMapNormForward2D   Perform cross channel normalization
%   Z = localMapNormForward2D(X, windowSize, alpha, beta, k) computes the
%   channel normalized version Z of X using the parameters specified by
%   windowSize, alpha, beta and k.
%
%   Inputs:
%   X - The input feature channels for a set of images. A (H)x(W)x(C)x(N) 
%       array.
%   windowSize - The number of channels to use for the normalization of 
%       each element.
%   alpha - Multiplier for the normalization term.
%   beta - Exponent for the normalization term.
%   k - Offset for the normalization term.
%
%   Output:
%   Z - The output feature channels for the images. A (H)x(W)x(C)x(N)
%       array.

%   Copyright 2016 The MathWorks, Inc.
Z = builtin('_xchannelnorm', X, windowSize, alpha, beta, k);