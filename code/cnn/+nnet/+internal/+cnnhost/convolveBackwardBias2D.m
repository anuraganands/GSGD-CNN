function dLossdB = convolveBackwardBias2D(dLossdZ)
% convolveBackwardBias2D   Backpropagate through a convolutional layer to
% get the derivative with respect to the bias.
%
% Input:
% dLossdZ - The derivative of the loss with respect to the output of the
% convolutional layer. A floor((H + 2*padHeight - R)/strideHeight + 1) x
% floor((W + 2*padWidth - S)/strideWidth + 1) x
% (K) x (N) array.
%
% Output:
% dLossdB - The derivative of the loss with respect to the bias. A
% (1)x(1)x(K)x(1) array.

%   Copyright 2015-2016 The MathWorks, Inc.

dLossdB = sum(sum(sum(dLossdZ,1),2),4);
end
