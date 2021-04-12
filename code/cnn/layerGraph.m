function lgraph = layerGraph(varargin)
% layerGraph   Create a layer graph
%
%   lgraph = layerGraph() will create an empty layer graph.
%
%   lgraph = layerGraph(larray) will create a layer graph from the layer
%   array larray. The layers in the resulting layer graph will be connected
%   one after the other, in the same way that they are in larray. Note that
%   all of the layers must have unique non-empty names.
%
%   lgraph = layerGraph(dagNet) will extract a LayerGraph object from the
%   DAGNetwork object dagNet. This is useful for transfer learning.
%
%   Example:
%       Create a layer graph to describe a network with a skip layer
%       connection:
%
%       layers = [
%           imageInputLayer([28 28 1], 'Name', 'input')
%           convolution2dLayer(5, 20, 'Name', 'conv_1')
%           reluLayer('Name', 'relu_1')
%           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_2')
%           reluLayer('Name', 'relu_2')
%           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_3')
%           reluLayer('Name', 'relu_3')
%           additionLayer(2, 'Name', 'add')
%           fullyConnectedLayer(10, 'Name', 'fc')
%           softmaxLayer('Name', 'softmax')
%           classificationLayer('Name', 'classoutput')];
%
%       lgraph = layerGraph(layers);
%
%       lgraph = connectLayers(lgraph, 'relu_1', 'add/in2');
%
%   See also nnet.cnn.LayerGraph.

%   Copyright 2017 The MathWorks, Inc.

if nargin == 1 && iInputIsDAGNetwork(varargin{1})
    internalLayerGraph = varargin{1}.getLayerGraph();
    layerMap = varargin{1}.getLayerMap();
    lgraph = iInternalToExternalLayerGraph(internalLayerGraph, layerMap);
else
    parser = iCreateParserForConstructor();
    parser.parse(varargin{:});
    inputArguments = iPostProcessParsingResultsForConstructor(parser.Results);
    
    lgraph = nnet.cnn.LayerGraph( ...
        inputArguments.Layers);
end

end

function tf = iInputIsDAGNetwork(input)
tf = isa(input, 'DAGNetwork');
end

function externalLayerGraph = iInternalToExternalLayerGraph( internalLayerGraph, layersMap )
externalLayers = iExternalLayers(internalLayerGraph.Layers, layersMap);
hiddenConnections = nnet.internal.cnn.util.internalToHiddenConnections(internalLayerGraph.Connections);
externalLayerGraph = nnet.cnn.LayerGraph(externalLayers, hiddenConnections);
end

function externalLayers = iExternalLayers(internalLayers, layersMap)
externalLayers = layersMap.externalLayers( internalLayers );
end

function p = iCreateParserForConstructor()

p = inputParser;

defaultLayers = nnet.cnn.layer.Layer.empty(0,1);

p.addOptional('Layers', defaultLayers, @(x)iValidateLayers(x));

end

function iValidateLayers(layers)
if(~isa(layers, 'nnet.cnn.layer.Layer'))
    error(message('nnet_cnn:layerGraph:InvalidLayerArray'));
end
end

function inputArguments = iPostProcessParsingResultsForConstructor(results)
inputArguments = struct;
inputArguments.Layers = nnet.internal.cnn.util.validateLayersForLayerGraph(results.Layers);
end