function dLossdX = softmaxBackward2D(Z, dLossdZ)
% softmaxBackward2D   Perform backpropagation for a softmax layer
%
% Inputs:
% Z - The output from the softmax layer. A (H)x(W)x(D)x(N) array.
% dLossdZ - The derivative of the loss function with respect to the output
% of the softmax layer. A (H)x(W)x(D)x(N) array.
%
% Output:
% dLossdX - The derivative of the loss function with respect to the input
% of the softmax layer. A (H)x(W)x(D)x(N) array.
%
% If H and W are anything other than 1, then each input 'pixel' is treated
% as a separate sample.

%   Copyright 2015-2016 The MathWorks, Inc.

dotProduct = sum(Z.*dLossdZ, 3);
dLossdX = dLossdZ - dotProduct;
dLossdX = dLossdX.*Z;

end
