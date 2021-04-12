function numOutputs = getNumOutputs(layer)
% TODO: This should be a property of the layer, but to avoid changing that 
% class, we have created a function

% Copyright 2017 The MathWorks, Inc.

layerClass = class(layer);
switch layerClass
    case 'Some kind of layer that doesn''t exist yet'
        % This case just illustrates what you would need to do to specify 
        % that a layer has two outputs.
        numOutputs = 2;
    case 'nnet.internal.cnn.layer.DepthSlice'
        % This layer has a variable number of outputs, so we return NaN.
        numOutputs = NaN;
    case 'nnet.internal.cnn.layer.MaxPooling2D'
        numOutputs = NaN;
    otherwise
        % If the layer is not one of the layers mentioned above, it only 
        % has one output.
        numOutputs = 1;
end
end