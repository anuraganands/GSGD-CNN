function layerIndex = validateNetworkLayerNameOrIndex(layerNameOrIndex, internalLayers, fname)
% Checks that layerNameOrIndex actually exists in the array of
% internalLayers. The input fname is the name of the function calls this
% validation routine. The output is the layer index.

%   Copyright 2016 The MathWorks, Inc.

if ischar(layerNameOrIndex)
    name = layerNameOrIndex;
    
    [layerIndex, layerNames] = nnet.internal.cnn.layer.Layer.findLayerByName(internalLayers, name);
    
    try
        % pretty print error message. will print available layer names in
        % case of a mismatch.
        validatestring(name, layerNames, fname, 'layer');
    catch Ex
        throwAsCaller(Ex);
    end
    
    % Only 1 match allowed. This is guaranteed during construction of SeriesNetwork.
    assert(numel(layerIndex) == 1);
        
else
    validateattributes(layerNameOrIndex, {'numeric'},...
        {'positive', 'integer', 'real', 'scalar', '<=', numel(internalLayers)}, ...
        fname, 'layer');
    layerIndex = layerNameOrIndex;
end