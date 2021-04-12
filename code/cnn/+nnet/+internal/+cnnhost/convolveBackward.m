function Z = convolveBackward(X, W, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    verticalStride, horizontalStride)
% Wrapper function to switch between bult-in conv (which is parallelized
% over the filters and images) and using conv2 (which is parallelized on
% the internal dimensions height and width) multiple times

%   Copyright 2017 The MathWorks, Inc.

% Filter
filterHeight = size(W,1);
filterWidth = size(W,2);

% Batch info
batchSize = size(W,4)*size(X,4);

% When we back-convolve with a stride > 1, rows and columns of zeros are
% added to the input (downsampling derivative is upsampling). When using
% conv2 on an input with lots of zeros, internal optimizations make conv2
% extremely fast (zeros are skipped saving internal computation).
%
% Thus, in the backward convolution case, we calculate the percentage of
% saved computation as function of the stride. We then put the stride to 1
% since convolution is always used with a stride = 1 when back convolving.

savedComputationBackward = (horizontalStride-1 + verticalStride-1) / (horizontalStride+verticalStride);
verticalStride = 1;
horizontalStride = 1;

% Get the function to use to do convolution
useConv2Loop = determineConvStrategy(filterHeight, filterWidth,...
    batchSize,...
    savedComputationBackward);

if useConv2Loop
    Z = iConvolveForward2D(X, W, ...
        padTop, padLeft, ...
        padBottom, padRight, ...
        verticalStride, horizontalStride);
else
    Z = nnet.internal.cnnhost.convolveForward2D(X, W, ...
        padTop, padLeft, ...
        padBottom, padRight, ...
        verticalStride, horizontalStride);
end
end

function tf = determineConvStrategy(filterHeight, filterWidth,...
    batchSize,...
    savedComputationBackward)
% Return a logical value indicating which strategy to use in order to compute backward convolution:
%   True - use conv2 in a loop. 
%   False - use the built-in version of conv2

% The rule of thumb is: if the saved computation is big enough (75% of the
% input slices are zeros) and the filter is big enough (when the filter is
% really tiny the builtin is always faster) use conv2.

% Alternatively, if the batchSize (numExamples*numFilters) is very small,
% using conv2 is still a win

tf = filterHeight > 8 && filterWidth > 8 &&  ...
    savedComputationBackward >= 0.75 || ...
    batchSize < 8;

end

function Z = iConvolveForward2D(X, W, ...
    padTop, padLeft, ...
    padBottom, padRight, ...
    strideHeight, strideWidth)
% convolveForward2D   Convolve input images with kernels
%
% Inputs:
% X - The input feature maps for a set of images. A (H)x(W)x(C)x(N) array.
% W - The kernels for convolution. A (R)x(S)x(C)x(K) array.
% padTop - Padding on the top.
% padLeft - Padding on the left.
% padBottom - Padding on the bottom.
% padRight - Padding on the right.
% strideHeight - The stride in the y direction.
% strideWidth - The stride in the x direction.
%
% Output:
% Z - The output feature maps for the images. A
% floor((H + padTop + padBottom - R)/strideHeight + 1) x
% floor((W + padLeft + padRight - S)/strideWidth + 1) x
% (C) x (N) array.
%
% This corresponds to the cuDNN function "cudnnConvolutionForward".

%   Copyright 2015-2017 The MathWorks, Inc.

imageHeight = size(X,1);
imageWidth =  size(X,2);
numInputMaps = size(X,3);
numExamples =  size(X,4);

% Apply the padding to the images if necessary
if (padTop > 0) || (padLeft > 0) || (padBottom > 0) || (padRight > 0)
    X = iPadArray(X, padTop, padLeft, padBottom, padRight);
end

filterHeight = size(W,1);
filterWidth = size(W,2);
assert(size(W,3) == numInputMaps, 'Kernel dim 3 does not match data');
numOutputMaps = size(W,4);

convolvedImageHeightWithoutStride = imageHeight + padTop + padBottom - filterHeight + 1;
convolvedImageWidthWithoutStride = imageWidth + padLeft + padRight - filterWidth + 1;

% Allocate memory for the output
Z = zeros(convolvedImageHeightWithoutStride, ...
    convolvedImageWidthWithoutStride, ...
    numOutputMaps, numExamples, 'like', X);

% Flip the kernel in every dimension
W = rot90(W,2);
for n = 1:numExamples
    for k = 1:numOutputMaps
        for c = 1:numInputMaps
            % Perform 3D convolution
            Z(:,:,k,n) = Z(:,:,k,n) + conv2(X(:,:,c,n), W(:,:,c,k), 'valid');
        end
    end
end

% Downsample the result to account for stride
Z = Z(1:strideHeight:end,1:strideWidth:end, :, :);
end

function Y = iPadArray(X, padTop, padLeft, padBottom, padRight)
paddedSize = size(X);
paddedSize(1) = paddedSize(1) + padTop + padBottom;
paddedSize(2) = paddedSize(2) + padLeft + padRight;
Y = zeros(paddedSize, 'like', X);
imageTop = padTop + 1;
imageBottom = padTop + size(X,1);
imageLeft = padLeft + 1;
imageRight = padLeft + size(X,2);
Y(imageTop:imageBottom, imageLeft:imageRight, :, :) = X;
end