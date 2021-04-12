function dLossdX = poolingAverageBackward2D(Z, dLossdZ, X, ...
    poolHeight, poolWidth, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    strideHeight, strideWidth, ...
    includePadding) %#ok<INUSL>
% poolingAverageBackward2D   Perform backpropagation for mean pooling
%
% Inputs:
% Z - The output from the pooling layer. This is not used here, but is
% dLossdZ - The derivative of the loss function with respect to the output
% of the pooling layer. A
% floor((H + padTop + padBottom - poolHeight)/strideHeight + 1) x
% floor((W + padLeft + padRight - poolWidth)/strideWidth + 1) x
% (C) x (N) array.
% X - The input to the pooling layer. A (H)x(W)x(C)x(N) array.
% added for consistency with the cuDNN interface.
% poolHeight - The height of a pooling region
% poolWidth - The width of a pooling region
% padTop - Padding on the top.
% padLeft - Padding on the left.
% padBottom - Padding on the bottom.
% padRight - Padding on the right.
% strideHeight - The stride in the y direction.
% strideWidth - The stride in the x direction.
% includePadding (optional) - Specifies if padding should be included in
% the average. The default is true.
%
% Output:
% dLossdX - The derivative of the loss function with respect to the input
% of the pooling layer. A (H)x(W)x(C)x(N) array.


%   Copyright 2015-2017 The MathWorks, Inc.

% Work out if we want to include padding.
if nargin < 12
    includePadding = true;
end

dLossdX = builtin('_meanpoolBackward', dLossdZ, [poolHeight poolWidth], ...
    [padTop padLeft padBottom padRight], [strideHeight strideWidth], ...
    [size(X,1), size(X,2)], includePadding);

end