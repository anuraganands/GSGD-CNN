function dLossdX = sigmoidBackward(Z, dLossdZ, X) %#ok<INUSD>
% sigmoidBackward   Perform backpropagation for a Sigmoid layer
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
%
% This corresponds to the cuDNN function "cudnnActivationBackward" with the
% neuron activation set to "CUDNN_ACTIVATION_SIGMOID".

%   Copyright 2015-2016 The MathWorks, Inc.

dZdX = Z .* (1 - Z);
dLossdX = dLossdZ .* dZdX;
end
