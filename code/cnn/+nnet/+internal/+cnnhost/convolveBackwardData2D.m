function dLossdA = convolveBackwardData2D(X, W, dLossdZ, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    strideHeight, strideWidth)
% convolveBackwardData2D   Backpropagate through a convolutional layer to
% get the derivative with respect to the input.
%
% Inputs:
% X - The input to the convolutional layer. We only pass this so that we
% can get its dimensions. An (H)x(W)x(C)x(N) array.
% W - The weights for the convolutional layer. An (R)x(S)x(C)x(K) array.
% dLossdZ - The derivative of the loss with respect to the output of the
% convolutional layer. A 
% floor((H + padTop + padBottom - R)/strideHeight + 1) x
% floor((W + padLeft + padRight - S)/strideWidth + 1) x
% (C) x (N) array.
% padTop - Padding on the top.
% padLeft - Padding on the left.
% padBottom - Padding on the bottom.
% padRight - Padding on the right.
% strideHeight - The stride in the y direction.
% strideWidth - The stride in the x direction.
%
% Output:
% dLossdA - The derivative of the loss with respect to the input of the
% convolutional layer. An (H)x(W)x(C)x(N) array.

%   Copyright 2015-2017 The MathWorks, Inc.

% The height and width of an input image. Note that this cannot be deduced
% from W and dLossdZ.
imageHeight = size(X,1);
imageWidth = size(X,2);

dLossdA = nnet.internal.cnnhost.convolveBackwardData2DCore(...
    [imageHeight, imageWidth], W, dLossdZ, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    strideHeight, strideWidth);

end
