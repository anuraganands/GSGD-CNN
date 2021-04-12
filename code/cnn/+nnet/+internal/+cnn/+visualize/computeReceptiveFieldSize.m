function receptiveFieldSize = computeReceptiveFieldSize(net, layerNumber)
% computeReceptiveFieldSize   Compute the receptive field sizes for a layer
%
%   receptiveFieldSize = computeReceptiveFieldSize(net, layerNumber)
%   takes a network net and an index layerNumber (which cannot be the first
%   or last layer) and returns the maximum receptive field size for that
%   layer.
%
%   The receptive field of a neuron in this context is the region of the
%   input image that the neuron can "see".

%   Copyright 2016-2017 The MathWorks, Inc.

% Get the input size
inputSize = net.Layers(1).InputSize;

% Initialise the receptive field size data. We store the indices for the
% beginning and end of each receptive field along both spatial dimensions.
receptiveFieldSizeData = struct;
receptiveFieldSizeData.YStart = 1:inputSize(1); 
receptiveFieldSizeData.YEnd = 1:inputSize(1);
receptiveFieldSizeData.XStart = 1:inputSize(2);
receptiveFieldSizeData.XEnd = 1:inputSize(2);

for i = 2:layerNumber
    currentLayer = net.Layers(i);
    layerType = class(currentLayer);
    switch layerType
        case 'nnet.cnn.layer.Convolution2DLayer'
            receptiveFieldSizeData = iGetReceptiveFieldSizeForConvolutionLayer( ...
                currentLayer, receptiveFieldSizeData);
        case 'nnet.cnn.layer.MaxPooling2DLayer'
            receptiveFieldSizeData = iGetReceptiveFieldSizeForMaxPoolingLayer( ...
                currentLayer, receptiveFieldSizeData);
        case 'nnet.cnn.layer.AveragePooling2DLayer'
            receptiveFieldSizeData = iGetReceptiveFieldSizeForAveragePoolingLayer( ...
                currentLayer, receptiveFieldSizeData);
        case 'nnet.cnn.layer.FullyConnectedLayer'
            receptiveFieldSizeData = iGetReceptiveFieldSizeForFullyConnectedLayer( ...
                currentLayer, receptiveFieldSizeData);
        otherwise
            % The layer does not alter the receptive field size. Do
            % nothing.
    end
end

% There can be a range of receptive field sizes due to padding. So we
% return the maximum size.
receptiveFieldHeights = iCalculateReceptiveFieldHeights(receptiveFieldSizeData);
receptiveFieldWidths = iCalculateReceptiveFieldWidths(receptiveFieldSizeData);
receptiveFieldSize = [ ...
    max(receptiveFieldHeights) ...
    max(receptiveFieldWidths) ...
    inputSize(3)
    ];

end

function receptiveFieldSizeData = iGetReceptiveFieldSizeForConvolutionLayer( ...
    layer, receptiveFieldSizeData)

[paddingSize, filterSize, stride] = iGetConvolutionLayerParameters(layer);

receptiveFieldSizeData = iPadReceptiveFields(receptiveFieldSizeData, paddingSize);
receptiveFieldSizeData = iApplyFilterSizeToReceptiveFields(receptiveFieldSizeData, filterSize);
receptiveFieldSizeData = iApplyStrideToReceptiveFields(receptiveFieldSizeData, stride);

end

function receptiveFieldSizeData = iGetReceptiveFieldSizeForMaxPoolingLayer( ...
    layer, receptiveFieldSizeData)

receptiveFieldSizeData = iGetReceptiveFieldSizeForPoolingLayer( ...
    layer, receptiveFieldSizeData);

end

function receptiveFieldSizeData = iGetReceptiveFieldSizeForAveragePoolingLayer( ...
    layer, receptiveFieldSizeData)

receptiveFieldSizeData = iGetReceptiveFieldSizeForPoolingLayer( ...
    layer, receptiveFieldSizeData);

end

function receptiveFieldSizeData = iGetReceptiveFieldSizeForFullyConnectedLayer( ...
    layer, receptiveFieldSizeData)

filterSize = iGetFullyConnectedFilterSize(layer);

receptiveFieldSizeData = iApplyFilterSizeToReceptiveFields(receptiveFieldSizeData, filterSize);

end

function receptiveFieldSizeData = iGetReceptiveFieldSizeForPoolingLayer( ...
    layer, receptiveFieldSizeData)

[paddingSize, poolSize, stride] = iGetPoolingLayerParameters(layer);

receptiveFieldSizeData = iPadReceptiveFields(receptiveFieldSizeData, paddingSize);
receptiveFieldSizeData = iApplyFilterSizeToReceptiveFields(receptiveFieldSizeData, poolSize);
receptiveFieldSizeData = iApplyStrideToReceptiveFields(receptiveFieldSizeData, stride);

end

function [paddingSize, filterSize, stride] = iGetConvolutionLayerParameters(layer)
paddingSize = layer.PaddingSize;
filterSize = layer.FilterSize;
stride = layer.Stride;
end

function [paddingSize, poolSize, stride] = iGetPoolingLayerParameters(layer)
paddingSize = layer.PaddingSize;
poolSize = layer.PoolSize;
stride = layer.Stride;
end

function filterSize = iGetFullyConnectedFilterSize(layer)
internalLayer = nnet.cnn.layer.Layer.getInternalLayers(layer);
internalLayer = internalLayer{1};
filterSize = internalLayer.InputSize(1:2);
end

function receptiveFieldSizeData = iPadReceptiveFields(receptiveFieldSizeData, paddingSize)

[ ...
    receptiveFieldSizeData.YStart, ...
    receptiveFieldSizeData.YEnd] = iPadReceptiveFieldsAlongOneDimension( ...
    receptiveFieldSizeData.YStart, ...
    receptiveFieldSizeData.YEnd, ...
    paddingSize(1:2));

[ ...
    receptiveFieldSizeData.XStart, ...
    receptiveFieldSizeData.XEnd] = iPadReceptiveFieldsAlongOneDimension( ...
    receptiveFieldSizeData.XStart, ...
    receptiveFieldSizeData.XEnd, ...
    paddingSize(3:4));

end

function [startIndices, endIndices] = iPadReceptiveFieldsAlongOneDimension( ...
    startIndices, endIndices, paddingSize)

% Note that this function assumes that the padding length is less than the 
% filter/pool size.
startPaddingArray = ones(1, paddingSize(1));
endPaddingArray = ones(1, paddingSize(2));

startIndices = [startIndices(1)*startPaddingArray startIndices startIndices(end)*endPaddingArray];
endIndices = [endIndices(1)*startPaddingArray endIndices endIndices(end)*endPaddingArray];

end

function receptiveFieldSizeData = iApplyFilterSizeToReceptiveFields(receptiveFieldSizeData, filterSize)

[ ...
    receptiveFieldSizeData.YStart, ...
    receptiveFieldSizeData.YEnd] = iApplyFilterSizeToReceptiveFieldsAlongOneDimension( ...
    receptiveFieldSizeData.YStart, ...
    receptiveFieldSizeData.YEnd, ...
    filterSize(1));

[ ...
    receptiveFieldSizeData.XStart, ...
    receptiveFieldSizeData.XEnd] = iApplyFilterSizeToReceptiveFieldsAlongOneDimension( ...
    receptiveFieldSizeData.XStart, ...
    receptiveFieldSizeData.XEnd, ...
    filterSize(2));

end

function [startIndices, endIndices] = iApplyFilterSizeToReceptiveFieldsAlongOneDimension( ...
    startIndices, endIndices, filterLength)

startIndices = startIndices(1:(end-filterLength+1));
endIndices = endIndices((filterLength):end);

end

function receptiveFieldSizeData = iApplyStrideToReceptiveFields(receptiveFieldSizeData, stride)

[ ...
    receptiveFieldSizeData.YStart, ...
    receptiveFieldSizeData.YEnd] = iApplyStrideToReceptiveFieldsAlongOneDimension( ...
    receptiveFieldSizeData.YStart, ...
    receptiveFieldSizeData.YEnd, ...
    stride(1));

[ ...
    receptiveFieldSizeData.XStart, ...
    receptiveFieldSizeData.XEnd] = iApplyStrideToReceptiveFieldsAlongOneDimension( ...
    receptiveFieldSizeData.XStart, ...
    receptiveFieldSizeData.XEnd, ...
    stride(2));

end

function [startIndices, endIndices] = iApplyStrideToReceptiveFieldsAlongOneDimension( ...
    startIndices, endIndices, strideLength)

numFields = length(startIndices);
lastValueToSample = iCalculateLastValueToSample(numFields, strideLength);
startIndices = startIndices(1:strideLength:lastValueToSample);
endIndices = endIndices(1:strideLength:lastValueToSample);

end

function lastValueToSample = iCalculateLastValueToSample(numFields, strideLength)
% This calculation is the same as the standard output size calculation.
outputSize = floor((numFields - 1)/strideLength) + 1;
lastValueToSample = outputSize*strideLength;
end

function heights = iCalculateReceptiveFieldHeights(receptiveFieldSizeData)
heights = receptiveFieldSizeData.YEnd - receptiveFieldSizeData.YStart + 1;
end

function widths = iCalculateReceptiveFieldWidths(receptiveFieldSizeData)
widths = receptiveFieldSizeData.XEnd - receptiveFieldSizeData.XStart + 1;
end