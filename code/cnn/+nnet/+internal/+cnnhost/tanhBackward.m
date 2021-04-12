function dLossdX = tanhBackward(Z, dLossdZ, X) %#ok<INUSD>
% tanhBackward   Perform backpropagation for a Sigmoid layer
%
% Inputs:
% Z - The output from the sigmoid layer.
% dLossdZ - The derivative of the loss function with respect to the output
% of the sigmoid layer. A (H)x(W)x(C)x(N) array.
% X - The input to the sigmoid layer. A (H)x(W)x(C)x(N) array. Unused, here
% to match cuDNN
%
% Output:
% dLossdX - The derivative of the loss function with respect to the input
% of the sigmoid layer. A (H)x(W)x(C)x(N) array.

%   Copyright 2015-2016 The MathWorks, Inc.

dLossdX = (1 - Z.^2) .* dLossdZ;
end
