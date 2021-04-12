function larray = validateLayersForLayerGraph(larray, existingLayers)
% validateLayersForLayerGraph - Validate layers array for addition to a
% layer graph
%
%   larray = validateLayersForLayerGraph(larray) takes an array of layers
%   larray and validates it for addition to a layer graph. The validated
%   array of layers is returned.
%
%   larray = validateLayersForLayerGraph(larray, existingLayers) also
%   accepts an array of layers existingLayers which represents the layers
%   already present in the layer graph.
%
%   Both larray and existingLayers are arrays whose elements are subclasses
%   of nnet.cnn.layer.Layer.

if ( nargin < 2 )
    existingLayers = nnet.cnn.layer.Layer.empty(0,1);
end

validateattributes(larray, {'nnet.cnn.layer.Layer'},{});

% Ensure that larray is a column vector.
larray = larray(:);

% larray must not contain a SequenceInputLayer or a LSTMLayer.
isSequenceInputLayer = arrayfun(@(x) iIsSequenceInputLayer(x), larray);
if any(isSequenceInputLayer)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:SequenceInputLayerNotAllowed'));
end
isRNNLayer = arrayfun(@(x) iIsRNNLayer(x), larray);
if any(isRNNLayer)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:RecurrentLayerNotAllowed'));
end

% If the existingLayers have an input (or output) layer already, then
% larray must not contain another input (or output) layer.
iAssertOnlyOneInputAndOutputLayer(larray, existingLayers);

iAssertNoInputLayersAfterFirstLayer(larray);

iAssertNoOutputLayersBeforeLastLayer(larray);

% All elements of larray must have non-empty and unique names and they
% should be different from the existing layer names.
iAssertUniqueAndNonEmptyLayerNames(larray, existingLayers);

% Assert that none of the names have the forward slash character, which is
% reserved for special use.
iAssertNoForwardSlashInLayerNames(larray);
end

function tf = iIsSequenceInputLayer(layer)
tf = isa(layer,'nnet.cnn.layer.SequenceInputLayer');
end

function tf = iIsRNNLayer(layer)
internalLayer = iGetInternalLayers(layer);
tf = isa(internalLayer{1},'nnet.internal.cnn.layer.Updatable');
end

function iAssertOnlyOneInputAndOutputLayer(layers, existingLayers)
internalLayers = iGetInternalLayers(layers);
haveInputLayer = iCheckForInputLayers(internalLayers);
haveOutputLayer = iCheckForOutputLayers(internalLayers);

if ( haveInputLayer || haveOutputLayer )
    existingInternalLayers = iGetInternalLayers(existingLayers);
    haveExistingInputLayer = iCheckForInputLayers(existingInternalLayers);
    haveExistingOutputLayer = iCheckForOutputLayers(existingInternalLayers);
    
    if ( haveInputLayer && haveExistingInputLayer )
        error(message('nnet_cnn:nnet:cnn:LayerGraph:OneInputLayerAllowed'));
    end
    if ( haveOutputLayer && haveExistingOutputLayer )
        error(message('nnet_cnn:nnet:cnn:LayerGraph:OneOutputLayerAllowed'));
    end
end
end

function iAssertUniqueAndNonEmptyLayerNames(layers, existingLayers)
layerNames = arrayfun(@(x) x.Name, layers, 'UniformOutput',false);
% Elements of layerNames cannot be empty.
isEmptyLayerName = cellfun(@(x) isempty(x), layerNames);
if any(isEmptyLayerName)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:LayerNamesMustBeNonEmpty'));
end
% Elements of layerNames must be unique.
uniqueLayerNames = unique(layerNames);
if ( numel(layerNames) ~= numel(uniqueLayerNames) )
    error(message('nnet_cnn:nnet:cnn:LayerGraph:LayerNamesMustBeUnique'));
end
% Elements of layerNames must not already exist in existingLayerNames.
existingLayerNames = arrayfun(@(x) x.Name, existingLayers, 'UniformOutput',false);
isExistingLayerName = ismember(layerNames,existingLayerNames);
if any(isExistingLayerName)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:LayerNamesMustNotAlreadyExist'));
end
end

function internalLayers = iGetInternalLayers(layers)
internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers(layers);
end

function haveInputLayers = iCheckForInputLayers(internalLayers)
haveInputLayers = any(cellfun(@(x) iIsInternalInputLayer(x), internalLayers));
end

function haveOutputLayers = iCheckForOutputLayers(internalLayers)
haveOutputLayers = any(cellfun(@(x) iIsInternalOutputLayer(x), internalLayers));
end

function tf = iIsInternalInputLayer(internalLayer)
tf = isa(internalLayer,'nnet.internal.cnn.layer.ImageInput');
end

function tf = iIsInternalOutputLayer(internalLayer)
tf = isa(internalLayer,'nnet.internal.cnn.layer.OutputLayer');
end

function iAssertNoForwardSlashInLayerNames(layers)
layerNames = arrayfun(@(x) x.Name, layers, 'UniformOutput',false);
numLayers = numel(layers);
for i = 1:numLayers
    iAssertNoForwardSlashInLayerName(layerNames{i});
end
end

function iAssertNoForwardSlashInLayerName(layerName)
if contains(layerName, '/')
    error(message( ...
        'nnet_cnn:nnet:cnn:LayerGraph:LayerNamesCannotContainForwardSlash', ...
        layerName));
end
end

function iAssertNoInputLayersAfterFirstLayer(layers)
if iInputLayersAfterFirstLayer(layers)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:LayersAfterTheFirstCannotBeInputLayers'));
end
end

function tf = iInputLayersAfterFirstLayer(layers)
tf = iThereIsMoreThanOneLayer(layers) && ...
    iAtLeastOneLayerIsAnInputLayer(layers(2:end));
end

function tf = iAtLeastOneLayerIsAnInputLayer(layers)
internalLayers =  nnet.cnn.layer.Layer.getInternalLayers(layers);
tf = any(cellfun(@(x)isa(x, 'nnet.internal.cnn.layer.ImageInput'), internalLayers));
end

function iAssertNoOutputLayersBeforeLastLayer(layers)
if iOutputLayersBeforeLastLayer(layers)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:LayersBeforeTheLastCannotBeOutputLayers'));
end
end

function tf = iOutputLayersBeforeLastLayer(layers)
tf = iThereIsMoreThanOneLayer(layers) && ...
    iAtLeastOneLayerIsAnOutputLayer(layers(1:end-1));
end

function tf = iAtLeastOneLayerIsAnOutputLayer(layers)
internalLayers =  nnet.cnn.layer.Layer.getInternalLayers(layers);
tf = any(cellfun(@(x)isa(x, 'nnet.internal.cnn.layer.OutputLayer'), internalLayers));
end

function tf = iThereIsMoreThanOneLayer(layers)
tf = length(layers) > 1;
end