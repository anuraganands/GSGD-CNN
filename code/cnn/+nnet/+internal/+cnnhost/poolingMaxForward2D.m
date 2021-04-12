function Z = poolingMaxForward2D(X, ...
    poolHeight, poolWidth, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    verticalStride, horizontalStride)
% poolingMaxForward2D   Forward max pooling on the host
%   Z = poolingMaxForward2D(X, poolHeight, poolWidth, verticalPad, horizontalPad, verticalStride, horizontalStride)
%   computes the max pooling Z of the input X using the pooling region 
%   size defined by poolHeight and poolWidth. Padding size is set with 
%   verticalPad and horizontalPad, and the vertical and horizontal stride 
%   are set with verticalStride and horizontalStride.
%
%   Inputs:
%   X - Input channels for a set of images. A (H)x(W)x(C)x(N) array.
%   poolHeight - The height of each pooling region
%   poolWidth - The width of each pooling region
%   padTop - Padding on the top.
%   padLeft - Padding on the left.
%   padBottom - Padding on the bottom.
%   padRight - Padding on the right.
%   verticalStride - The vertical stride.
%   horizontalStride - The horizontal stride.
%
%   Output:
%   Z - The output feature channels for the images. A
%       floor((H + padTop + padBottom - poolHeight)/verticalStride + 1) x
%       floor((W + padLeft + padRight - poolWidth)/horizontalStride + 1) x
%       (C) x (N) array.

%   Copyright 2016-2017 The MathWorks, Inc.

Z = builtin('_maxpool', X, [poolHeight poolWidth], ...
   [padTop padLeft padBottom padRight], ...
   [verticalStride horizontalStride]);