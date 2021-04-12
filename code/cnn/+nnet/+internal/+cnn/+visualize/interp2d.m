function outputImage = interp2d(I,X,Y,fillValues, varargin)%#codegen
% FOR INTERNAL USE ONLY -- This function is intentionally
% undocumented and is intended for use only within other toolbox
% classes and functions. Its behavior may change, or the feature
% itself may be removed in a future release.
%
% Vq = INTERP2D(V,XINTRINSIC,YINTRINSIC,FILLVAL, SmoothEdges)
% computes 2-D interpolation on the input grid V at locations in the
% intrinsic coordinate system XINTRINSIC, YINTRINSIC. The value of the
% output grid Vq(I,J) is determined by performing 2-D interpolation at
% locations specified by the corresponding grid locations in
% XINTRINSIC(I,J), YINTRINSIC(I,J). XINTRINSIC and YINTRINSIC are plaid
% matrices of the form constructed by MESHGRID. When V has more than two
% dimensions, the output Vq is determined by interpolating V a slice at a
% time beginning at the 3rd dimension.

% Copyright 2012-2016 The MathWorks, Inc.

% Algorithm Notes
%
% This function is intentionally very similar to the MATLAB INTERP2
% function. The differences between INTERP2 and this function are:
%
% 1) Edge behavior. This function uses the 'fill' pad method described in
% the help for makeresampler. When the interpolation kernel partially
% extends beyond the grid, the output value is determined by blending fill
% values and input grid values.
% This behavior is on by default, unless SmoothEdges is specified and set
% to false.
%
% 2) Plane at a time behavior. When the input grid has more than 2
% dimensions, this function treats the input grid as a stack of 2-D
% interpolation problems beginning at the 3rd dimension. Also handles > 3
% dims by stacking along 3rd dim.
%
% 3) Degenerate 2-D grid behavior. Unlike interp2, this function handles
% input grids that are 1-by-N or N-by-1.


% Stack all 2-D planes along third dim.
[M,N,numChannels,numObs] = size(I);
inputShape = [M,N,numChannels, numObs];
if numObs > 1
    % reshape input so that dim > 3 are stacked along third dim.
    inputImage = reshape(I, inputShape(1), inputShape(2), prod(inputShape(3:end)));
else
    inputImage = I;
end

if(islogical(inputImage))
    inputImage = uint8(inputImage);
end

% MATLAB interp2 requires floats. Cast all non-floats to single.
if ~isfloat(inputImage)
    inputImage = single(inputImage);
    X = single(X);
    Y = single(Y);
end
    
fillValues = cast(fillValues,'like', inputImage);

if (~ismatrix(inputImage) && isscalar(fillValues))
    % If we are doing plane at at time behavior, make sure fillValues
    % always propogates through code as a matrix of size determine by
    % dimensions 3:end of inputImage.
    sizeInputImage = size(inputImage);
    if (ndims(inputImage)==3)
        % This must be handled as a special case because repmat(X,N)
        % replicates a scalar X as a NxN matrix. We want a Nx1 vector.
        sizeVec = [sizeInputImage(3) 1];
    else
        sizeVec = sizeInputImage(3:end);
    end
    fillValues = repmat(fillValues,sizeVec);
end

% Preallocate outputImage so that we can call interp2 a plane at a time if
% the number of dimensions in the input image is greater than 2.
if ~ismatrix(inputImage)
    [~,~,P] = size(inputImage);
    sizeInputVec = size(inputImage);
    outputImage = zeros([size(X) sizeInputVec(3:end)],'like',inputImage);
else
    P = 1;
    outputImage = zeros(size(X),'like',inputImage);
end

[inputImage,X,Y] = iPadImage(inputImage,X,Y,fillValues);

for plane = 1:P
    outputImage(:,:,plane) = interp2(inputImage(:,:,plane),X,Y,'linear',fillValues(plane));
end

% reshape output image to match dims(3:end) of input.
outputShape = size(outputImage);
outputShape = [outputShape(1:2) inputShape(3:end)];
outputImage = reshape(outputImage, outputShape);

outputImage = cast(outputImage,'like', I);

%--------------------------------------------------------------------------
function [paddedImage,X,Y] = iPadImage(inputImage,X,Y,fillValues)
% We achieve the 'fill' pad behavior from makeresampler by prepadding our
% image with the fillValues and translating our X,Y locations to the
% corresponding locations in the padded image. We pad two elements in each
% dimension to account for the limiting case of bicubic interpolation,
% which has a interpolation kernel half-width of 2.

pad = 3;
X = X+pad;
Y = Y+pad;

sizeInputImage = size(inputImage);
sizeOutputImage = sizeInputImage;
sizeOutputImage(1:2) = sizeOutputImage(1:2) + [2*pad 2*pad];

if isscalar(fillValues)
    paddedImage = repmat(fillValues,sizeOutputImage);
    if(ismatrix(inputImage))
        paddedImage(4:end-3,4:end-3,:) = inputImage;
    else
        for pInd = 1:prod(sizeOutputImage(3:end))
            paddedImage(4:end-3,4:end-3,pInd) = inputImage(:,:,pInd);
        end
    end

else
    if islogical(inputImage)
        paddedImage = false(sizeOutputImage);
    else
        paddedImage = zeros(sizeOutputImage,'like',inputImage);
    end
    [~,~,numPlanes] = size(inputImage);
    for i = 1:numPlanes
        paddedImage(:,:,i) = iPadArray(inputImage(:,:,i),[pad pad],fillValues(i));
    end
    
end   

%--------------------------------------------------------------------------
function out = iPadArray(I, padSize, padVal)
% pad image along first 2 dims using padarray's 'both' style using constant
% pad value.

assert(ismatrix(I));

% pad top and bottom
N = size(I,2);
padding = repelem(padVal, padSize(1), N);
out = [padding; I; padding];

% pad right and left
M = size(out,1);
padding = repelem(padVal, M, padSize(2));
out = [padding out padding];



