function dLossdW = convolveBackwardFilter2D(X, W, dLossdZ, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    strideHeight, strideWidth)
% convolveBackwardFilter2D   Backpropagate through a convolutional layer to
% get the derivative with respect to the filters.
%
% Inputs:
% X - The input to the convolutional layer. An (H)x(W)x(C)x(N) array.
% W - The filters for the convolutional layer. We only pass these so that
% we can get their dimensions. An (R)x(S)x(C)x(K) array.
% dLossdZ - The derivative of the loss with respect to the output of the
% convolutional layer. An (H-R+1)x(W-S+1)x(K)x(N) array.
% padTop - Padding on the top.
% padLeft - Padding on the left.
% padBottom - Padding on the bottom.
% padRight - Padding on the right.
% strideHeight - The stride in the y direction.
% strideWidth - The stride in the x direction.
%
% Output:
% dLossdW - The derivative of the loss with respect to the filters. An
% (R)x(S)x(C)x(K) array.

%   Copyright 2016-2017 The MathWorks, Inc.

% The height and width of the filters. Note that this cannot be deduced
% from dLossdZ and X.

backwardFilterConvolution = true;
dLossdW = builtin('_batchconvBackward', X, dLossdZ, ...
    [padTop padLeft padBottom padRight], [strideHeight strideWidth],...
    [size(W,1), size(W,2)], backwardFilterConvolution);
end



