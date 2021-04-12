function Z = stridedConv(X, W, padTop, padLeft, padBottom, padRight, strideHeight, strideWidth)
% Call the internal strided conv matlab.internal.math.stridedConvn. 
% Inputs:
% X - The input to the convolutional layer. An (H)x(W)x(C)x(N) array.
% W - The filters for the convolutional layer.  An (R)x(S)x(C)x(K) array.
% padTop, padBottom - Vertical padding.
% padLeft, padRight - Horizontal padding.
% strideHeight, strideWidth - Vertical/Horizontal stride (optional, 1 by default).
% skipZeroChecks - Skip checks used to optimize cases where inputs have many zeros.
%
% Outputs:
% Z - Convolution between X and W. An (R)x(S)x(K)x(N) array.

% Copyright 2017 The MathWorks, Inc.

    if nargin < 8
        strideHeight = 1;
        strideWidth = 1;
    end
    
    Z = builtin('_batchconv', X, W, [padTop padLeft padBottom padRight], ...
        [strideHeight strideWidth]);
    
end
