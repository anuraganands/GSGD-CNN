classdef LayerGraph
    % LayerGraph   A graph consisting of network layers
    %
    %   A LayerGraph object is used to describe the layout of a Directed
    %   Acyclic Graph (DAG) network. You can add layers to the graph and
    %   connect them together.    
    %
    %   LayerGraph properties:
    %       Layers              - The layers of the network
    %       Connections         - The connections between the layers
    %
    %   LayerGraph methods:
    %       addLayers           - Add layers to the LayerGraph
    %       connectLayers       - Connect layers in the LayerGraph
    %       removeLayers        - Remove layers from the LayerGraph
    %       disconnectLayers    - Disconnect layers in the LayerGraph
    %       plot                - Plot a diagram of the LayerGraph
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
    %   See also Layer.
    
    %   Copyright 2016-2018 The MathWorks, Inc.
    
    properties(Dependent, SetAccess = private)
        % Layers   The layers in the layer graph
        %   An array of the layers in the layer graph. Each layer has 
        %   different properties depending on what type of layer it is.
        Layers
        
        % Connections   A table of connections between the layers
        %   A table with one row for each connection between two layers. 
        %   The table has two columns:
        %       Source          - The name of the layer (and the layer 
        %                         output if applicable) where the 
        %                         connection begins.
        %       Destination     - The name of the layer (and the layer 
        %                         input if applicable) where the connection
        %                         ends.
        Connections
    end
    
    properties(Dependent, Hidden)
        % HiddenConnections
        HiddenConnections
    end
    
    methods
        function val = get.Layers(this)
            val = this.PrivateDirectedGraph.Nodes.Layers;
        end
        
        function val = get.Connections(this)
            internalConnections = nnet.internal.cnn.util.hiddenToInternalConnections( ...
                this.PrivateDirectedGraph.Edges);
            val = nnet.internal.cnn.util.internalToExternalConnections( ...
                internalConnections, ...
                this.Layers);
        end
        
        function val = get.HiddenConnections(this)
            val = this.PrivateDirectedGraph.Edges;
        end
    end
    
    properties(Access = private)
        % PrivateDirectedGraph   A directed graph object
        PrivateDirectedGraph
        
        % Sizes   The output sizes for each layer
        Sizes
        
        % TopologicalOrder   Original indices of layers. This property
        % is empty initially but after calling toposort, this is filled
        % with indices of original layers when the layers are arranged in a
        % topologically sorted order.
        TopologicalOrder
    end
    
    properties(Dependent, Access = private)
        % NumLayers   The number of layers in the layer graph
        NumLayers
    end
    
    methods
        function val = get.NumLayers(this)
            val = numel(this.PrivateDirectedGraph.Nodes);
        end
    end
    
    methods(Hidden)
        function this = LayerGraph(larray, connections)
            % LayerGraph   Constructor for layer graph object

            if nargin == 1
                this.PrivateDirectedGraph = iCreateGraphFromLayerArray(larray);
            else
                this.PrivateDirectedGraph = iCreateGraphFromLayerArrayAndConnectionsTable( ...
                    larray, connections);
            end
        end
        
        function graph = extractPrivateDirectedGraph(this)
            % extractPrivateDirectedGraph   
            graph = this.PrivateDirectedGraph;
        end
        
        function this = setTopologicalOrder(this, order)
            % Set the TopologicalOrder property.
            this.TopologicalOrder = order;
        end
        function this = toposort(this)
            [sortedIndices, sortedGraph] = this.PrivateDirectedGraph.toposort('Order','stable');
            this.PrivateDirectedGraph = sortedGraph;
            this.TopologicalOrder = sortedIndices;
            if(~isempty(this.Sizes))
                this.Sizes = this.Sizes(sortedIndices);
            end
        end
        
        function this = setLayers(this, externalLayers)
            this.PrivateDirectedGraph.Nodes.Layers = externalLayers;
        end
        
        function sizes = extractSizes(this)
            % extractSizes
            sizes = this.Sizes;
        end
        function this = setSizes(this, sizes)
            % Set the Sizes property.
            this.Sizes = sizes;
        end
        
        function this = inferParameters(this)
            [~, analysis] = nnet.internal.cnn.layer.util.inferParameters(this);
            this = analysis.LayerGraph;
        end
        
        function topologicalOrder = extractTopologicalOrder(this)
            topologicalOrder = this.TopologicalOrder;
        end
    end
   
    methods
        function this = addLayers(this, larray)
            % addLayers   Add layers to the LayerGraph object
            %
            %   lgraph = addLayers(lgraph, larray) will add the layers in 
            %   larray to lgraph, where larray is an array of Layer 
            %   objects. The layers from larray will be connected 
            %   sequentially. 
            
            existingLayers = this.Layers;
            larray = iValidateLayers(larray, existingLayers);
            
            % Create tables
            layerTable = iLayerTable(larray);
            numOldLayers = this.NumLayers;
            numNewLayers = numel(larray);
            firstNewIndex = numOldLayers + 1;
            lastNewIndex = numNewLayers + numOldLayers;
            edgeList = iGenerateSeriesEdgeList(firstNewIndex, lastNewIndex);
            portList = ones(size(edgeList));
            edgeTable = iEdgeTable(edgeList, portList);
            
            % Add the nodes and edges to the graph
            this.PrivateDirectedGraph = this.PrivateDirectedGraph.addnode(layerTable);
            this.PrivateDirectedGraph = this.PrivateDirectedGraph.addedge(edgeTable);
            
            iWarnIfThereIsNoReturnArgument(nargout);
        end
        
        function this = removeLayers(this, layerNames)
            % removeLayers Remove layers from the LayerGraph object
            %
            %   lgraph = removeLayers(lgraph, layerNames) removes the 
            %   layers specified by layerNames from lgraph. layerNames is 
            %   either the name of a layer or a cell array where each entry
            %   is a layer name.
            
            %   Note that removing a layer will remove all ingoing and 
            %   outgoing connections for that layer.
            
            iValidateInputForRemoveLayers(layerNames);
            
            % If the input is not a cell, wrap it in a cell.
            if ~iscell(layerNames)
                layerNames = {layerNames};
            end
            
            % Remove duplicate names
            layerNames = unique(layerNames);
            
            numLayersToRemove = numel(layerNames);
            layerIndices = zeros(numLayersToRemove,1);
            for i = 1:numLayersToRemove
                iValidateLayerName( ...
                    layerNames{i}, ...
                    this.PrivateDirectedGraph.Nodes.Layers);
                layerIndices(i) = iConvertLayerNameToIndex( ...
                    layerNames{i}, this.PrivateDirectedGraph.Nodes.Layers);
            end
            
            this.PrivateDirectedGraph = this.PrivateDirectedGraph.rmnode( ...
                layerIndices);
            
            iWarnIfThereIsNoReturnArgument(nargout);
        end
        
        function this = connectLayers(this, s, d)
            % connectLayers   Connect layers in a layer graph
            %
            %   lgraph = connectLayers(lgraph, s, d) connects a source 
            %   layer to a destination layer in a layer graph, where s and
            %   d are character arrays.
            %     - If the source layer has a single output, then s is the
            %       name of the layer. If the source layer has multiple 
            %       outputs, then s is the name of the layer followed by 
            %       the '/' character, followed by the name of the layer 
            %       output.
            %     - If the destination layer has a single input, then d is
            %       the name of the layer. If the destination layer has
            %       multiple inputs, then d is the name of the layer
            %       followed by the '/' character, followed by the name of
            %       the layer input.
            
            iValidateConnectionSourceIsCorrectType(s);
            iValidateConnectionDestinationIsCorrectType(d);
            
            [startLayerName,startLayerIndex,~, layerOutputIndex] = ...
                iGetSourceInformation(s, this.PrivateDirectedGraph.Nodes.Layers);
            
            [endLayerName,endLayerIndex,~, layerInputIndex] = ...
                iGetDestinationInformation(d, this.PrivateDirectedGraph.Nodes.Layers);
            
            startLayer = this.PrivateDirectedGraph.Nodes.Layers(startLayerIndex);
            endLayer = this.PrivateDirectedGraph.Nodes.Layers(endLayerIndex);
            
            iThrowErrorIfStartLayerIsOutputLayer(startLayer);
            iThrowErrorIfEndLayerIsInputLayer(endLayer);
            iThrowErrorIfStartAndEndLayerAreSame(startLayerName, endLayerName);
            
            % Find the edge in the edge table. This will return 0 if the
            % edge  does not exist.
            edgeIndex = findedge(this.PrivateDirectedGraph, startLayerIndex, endLayerIndex);
            
            if iEdgeDoesNotExistYet(edgeIndex)
                % Get the layer input list for the end layer. Throw an 
                % error if the input is occupied.
                endLayerInputIndexList = iGetOccupiedLayerInputsForThisLayer( ...
                    this.PrivateDirectedGraph.Edges, endLayerIndex);
                iValidateEndLayerInputIsNotOccupied( ...
                    endLayerInputIndexList, layerInputIndex, d);
                
                this.PrivateDirectedGraph = ...
                    this.PrivateDirectedGraph.addedge( ...
                    iEdgeTable([startLayerIndex endLayerIndex], [layerOutputIndex layerInputIndex]) );
            else
                % Get the existing layer outputs and inputs list for this 
                % edge.
                layerOutputInputList = this.PrivateDirectedGraph.Edges.EndPorts{edgeIndex};
                
                % If the exact connection that is being added exists, throw
                % a specific error.
                iValidateThatConnectionDoesNotExist( ...
                    layerOutputInputList, ...
                    layerOutputIndex, ...
                    layerInputIndex);
                
                % Get the layer input list for the end layer. Throw an 
                % error if the input is occupied.
                endLayerInputIndexList = iGetOccupiedLayerInputsForThisLayer( ...
                    this.PrivateDirectedGraph.Edges, endLayerIndex);
                iValidateEndLayerInputIsNotOccupied( ...
                    endLayerInputIndexList, layerInputIndex, d);
                
                layerOutputInputList = [layerOutputInputList; layerOutputIndex layerInputIndex];
                
                this.PrivateDirectedGraph.Edges.EndPorts{edgeIndex} = ...
                    iSortPortListByInputPort(layerOutputInputList);
            end
            
            iWarnIfThereIsNoReturnArgument(nargout);
        end
        
        function this = disconnectLayers(this, s, d)
            % disconnectLayers   Disconnect layers in a layer graph
            %
            %   lgraph = disconnectLayers(lgraph, s, d) will disconnect a
            %   source layer from a destination layer in a layer graph,
            %   where s and d are character arrays described below:
            %     - If the source layer has a single output, then s is the
            %       name of the layer. If the source layer has multiple 
            %       outputs, then s is the name of the layer followed by 
            %       the '/' character, followed by the name of the layer 
            %       output.
            %     - If the destination layer has a single input, then d is
            %       the name of the layer. If the destination layer has
            %       multiple inputs, then d is the name of the layer
            %       followed by the '/' character, followed by the name of
            %       the layer input.
            
            iValidateConnectionSourceIsCorrectType(s);
            iValidateConnectionDestinationIsCorrectType(d);
            
            [~,startLayerIndex,~,layerOutputIndex] = ...
                iGetSourceInformation(s, this.PrivateDirectedGraph.Nodes.Layers);
            
            [~,endLayerIndex,~,layerInputIndex] = ...
                iGetDestinationInformation(d, this.PrivateDirectedGraph.Nodes.Layers);
            
            % Find the edge in the edge table. This will return 0 if the
            % edge does not exist.
            edgeIndex = findedge(this.PrivateDirectedGraph, startLayerIndex, endLayerIndex);
            
            if iEdgeExists(edgeIndex)
                % Get the existing layer outputs and inputs list for this 
                % edge.
                layerOutputInputList = this.PrivateDirectedGraph.Edges.EndPorts{edgeIndex};
                
                % Remove the connection from the port list if it's there
                if iConnectionExists( ...
                    layerOutputInputList, ...
                    layerOutputIndex, ...
                    layerInputIndex)
                    indicesForRowToRemove = ismember( ...
                        layerOutputInputList, [layerOutputIndex layerInputIndex], 'rows');
                    layerOutputInputList(indicesForRowToRemove,:) = [];
                end
                
                this.PrivateDirectedGraph.Edges.EndPorts{edgeIndex} = ...
                    layerOutputInputList;
                
                % If the port list is now empty, then we need to remove the
                % edge from the graph.
                if isempty(layerOutputInputList)
                    this.PrivateDirectedGraph = ...
                        this.PrivateDirectedGraph.rmedge(startLayerIndex, endLayerIndex);
                end
            end
            
            iWarnIfThereIsNoReturnArgument(nargout);
        end
        
        function plot(this)
            % plot   Plot a diagram of the layer graph
            %
            %   plot(lgraph) plots a diagram of the layer graph lgraph.
            %   Each layer in the diagram is labelled by its name.
            
            nodeNames = iGetLayerNames(this.PrivateDirectedGraph.Nodes.Layers);
            plot(this.PrivateDirectedGraph, 'NodeLabel', nodeNames, 'Layout', 'layered');
        end
        
        function out = saveobj(this)
            out.Version = 1.0;
            out.Layers = this.Layers;
            out.Connections = this.Connections;
        end
    end
    
    methods(Static)
        function this = loadobj(in)
            internalConnections = nnet.internal.cnn.util.externalToInternalConnections(in.Connections, in.Layers);
            hiddenConnections = nnet.internal.cnn.util.internalToHiddenConnections(internalConnections);
            this = nnet.cnn.LayerGraph(in.Layers, hiddenConnections);
        end
    end
end

function graph = iCreateGraphFromLayerArray(larray)

% Create tables
layerTable = iLayerTable(larray);
numLayers = numel(larray);
edgeList = iGenerateSeriesEdgeList(1,numLayers);
portList = ones(size(edgeList));
edgeTable = iEdgeTable(edgeList, portList);

% Create graph
graph = digraph(edgeTable, layerTable);

end

function graph = iCreateGraphFromLayerArrayAndConnectionsTable( ...
    larray, connections)
graph = digraph(connections, iLayerTable(larray));
end

function layerTable = iLayerTable(larray)
layerTable = table(larray, 'VariableNames', {'Layers'});
end

function edgeTable = iEdgeTable(edgeList, portList)
portList = iConvertMatrixToCellOfRows(portList);
edgeTable = table(edgeList, portList, 'VariableNames', {'EndNodes','EndPorts'});
end

function cellOfRows = iConvertMatrixToCellOfRows(matrix)
numRows = size(matrix,1);
cellOfRows = mat2cell(matrix, ones(1,numRows));
end

function endNodes = iGenerateSeriesEdgeList(firstIndex, lastIndex)
% iGenerateSeriesEdgeList   Create edge list for connecting layers in series
endNodes = [(firstIndex:(lastIndex-1))' ((firstIndex+1):lastIndex)'];
end

function larray = iValidateLayers(larray, existingLayers)
larray = nnet.internal.cnn.util.validateLayersForLayerGraph(larray, existingLayers);
end

% Start of layer input/output related code (also applicable to DAGNetwork)

function tf = iEdgeDoesNotExistYet(edgeIndex)
tf = (edgeIndex == 0);
end

% Functions common to nnet.cnn.layer.Layer

function names = iGetLayerNames(layers)
names = arrayfun(@(layer)layer.Name, layers, 'UniformOutput', false);
end

% Helpers for connecting, disconnecting and removing layers.
function [startLayerName,startLayerIndex,layerOutputName,layerOutputIndex] = iGetSourceInformation(s, layers)
iValidateThereAreNotMultipleForwardSlashesInSource(s);
sSplit = strsplit(s, '/');
startLayerName = sSplit{1};
iValidateLayerName( startLayerName, layers );
startLayerIndex = iConvertLayerNameToIndex( startLayerName, layers );
if(numel(sSplit) == 2)
    layerOutputName = sSplit{2};
else
    iThrowErrorIfLayerHasMultipleOutputs( layers(startLayerIndex) );
    layerOutputName = 'out';
end
layerOutputIndex = iGetLayerOutputIndex( layers(startLayerIndex), layerOutputName );
end

function [endLayerName,endLayerIndex,layerInputName,layerInputIndex] = iGetDestinationInformation(d, layers)
iValidateThereAreNotMultipleForwardSlashesInDestination(d);
tSplit = strsplit(d, '/');
endLayerName = tSplit{1};
iValidateLayerName( endLayerName, layers );
endLayerIndex = iConvertLayerNameToIndex( endLayerName, layers );
if(numel(tSplit) == 2)
    layerInputName = tSplit{2};
else
    iThrowErrorIfLayerHasMultipleInputs( layers(endLayerIndex) );
    layerInputName = 'in';
end
layerInputIndex = iGetLayerInputIndex( layers(endLayerIndex), layerInputName );
end

function iValidateThereAreNotMultipleForwardSlashesInSource(s)
if iContainsMoreThanOneBackslash(s)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:CannotHaveMultipleForwardSlashesInSource'));
end
end

function iValidateThereAreNotMultipleForwardSlashesInDestination(d)
if iContainsMoreThanOneBackslash(d)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:CannotHaveMultipleForwardSlashesInDestination'));
end
end

function tf = iContainsMoreThanOneBackslash(x)
tf = length(strfind(x, '/')) > 1;
end

function layerIndex = iConvertLayerNameToIndex(layerName, externalLayers)
layerNames = {externalLayers.Name}';
layerIndex = find(strcmp(layerNames, layerName));
end

function layerOutputIndex = iGetLayerOutputIndex(layer, layerOutputName)
layerClass = class(layer);
switch layerClass
    case 'nnet.cnn.layer.MaxPooling2DLayer'
        maxPoolingOutputNames = {'out', 'indices', 'size'};
        iValidateLayerOutputName(layer.Name, layerOutputName, maxPoolingOutputNames);
        if ~layer.HasUnpoolingOutputs && ~strcmp(maxPoolingOutputNames{1}, layerOutputName)
            error(message( ...
                'nnet_cnn:nnet:cnn:LayerGraph:MultipleOutputsForMaxPooling2DLayerWithNoUnpoolingOutputs', ...
                layer.Name, layerOutputName));
        end
        layerOutputIndex = find(strcmp(maxPoolingOutputNames, layerOutputName));
    otherwise
        iValidateLayerOutputName(layer.Name, layerOutputName, 'out');
        layerOutputIndex = 1;
end
end

function layerInputIndex = iGetLayerInputIndex(layer, layerInputName)
layerClass = class(layer);
switch layerClass
    case 'nnet.cnn.layer.MaxUnpooling2DLayer'
        maxUnpoolingInputNames = {'in','indices','size'};
        iValidateLayerInputName(layer.Name, layerInputName, maxUnpoolingInputNames);
        layerInputIndex = find(strcmp(maxUnpoolingInputNames, layerInputName));
    case 'nnet.cnn.layer.AdditionLayer'
        additionLayerNames = iGenerateNumericInputNames(layer.NumInputs);
        iValidateLayerInputName(layer.Name, layerInputName, additionLayerNames);
        layerInputIndex = str2double(layerInputName(3:end));
    case 'nnet.cnn.layer.DepthConcatenationLayer'
        depthConcatenationLayerNames = iGenerateNumericInputNames(layer.NumInputs);
        iValidateLayerInputName(layer.Name, layerInputName, depthConcatenationLayerNames);
        layerInputIndex = str2double(layerInputName(3:end));
    case 'nnet.cnn.layer.Crop2DLayer'
        crop2DInputNames = {'in','ref'};
        iValidateLayerInputName(layer.Name, layerInputName, crop2DInputNames);
        layerInputIndex = find(strcmp(crop2DInputNames, layerInputName));
    otherwise
        iValidateLayerInputName(layer.Name, layerInputName, 'in');
        layerInputIndex = 1;
end
end

function outputMatrix = iSortPortListByInputPort(inputMatrix)
[~, sortedIndices] = sort(inputMatrix(:,1));
outputMatrix = inputMatrix(sortedIndices,:);
end

function iThrowErrorIfStartLayerIsOutputLayer(startLayer)
if iLayerIsOutputLayer(startLayer)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:CannotConnectFromOutputLayer', startLayer.Name));
end
end

function tf = iLayerIsOutputLayer(layer)
internalLayer =  nnet.cnn.layer.Layer.getInternalLayers(layer);
tf = isa(internalLayer{:}, 'nnet.internal.cnn.layer.OutputLayer');
end

function iThrowErrorIfEndLayerIsInputLayer(endLayer)
if iLayerIsInputLayer(endLayer)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:CannotConnectToInputLayer', endLayer.Name));
end
end

function tf = iLayerIsInputLayer(layer)
internalLayer = nnet.cnn.layer.Layer.getInternalLayers(layer);
tf = isa(internalLayer{:}, 'nnet.internal.cnn.layer.ImageInput');
end

function iValidateLayerName(layerName, externalLayers)
if ~iLayerNameExists(layerName, externalLayers)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:LayerDoesNotExist', layerName))
end
end

function tf = iLayerNameExists(layerName, externalLayers)
layerNames = {externalLayers.Name}';
tf = any(strcmp(layerNames, layerName));
end

function iThrowErrorIfStartAndEndLayerAreSame(startLayerName, endLayerName)
if strcmp(startLayerName, endLayerName)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:CannotConnectLayerToItself', endLayerName));
end
end

function layerInputIndexList = iGetOccupiedLayerInputsForThisLayer(edgeTable, layerIndex)
layerOutputInputTable = edgeTable((edgeTable.EndNodes(:,2) == layerIndex),{'EndPorts'});
layerOutputInputList = cell2mat(layerOutputInputTable.EndPorts);
if(isempty(layerOutputInputList))
    layerInputIndexList = [];
else
    layerInputIndexList = layerOutputInputList(:,2);
end
end

function iValidateEndLayerInputIsNotOccupied(endLayerInputIndexList, layerInputIndex, destination)
if iLayerInputIsOccupied(endLayerInputIndexList, layerInputIndex)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:LayerInputIsNotFree', destination));
end
end

function tf = iLayerInputIsOccupied(layerInputIndexList, layerInputIndex)
tf = any(ismember(layerInputIndexList, layerInputIndex));
end

function iThrowErrorIfLayerHasMultipleInputs(layer)
if iLayerHasMultipleInputs(layer)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:MustSpecifyInputForMultipleInputLayer', layer.Name));
end
end

function tf = iLayerHasMultipleInputs(layer)
tf = iGetNumberOfInputsForLayer(layer) > 1;
end

function iThrowErrorIfLayerHasMultipleOutputs(layer)
if iLayerHasMultipleOutputs(layer)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:MustSpecifyOutputForMultipleOutputLayer', layer.Name));
end
end

function tf = iLayerHasMultipleOutputs(layer)
tf = iGetNumberOfOutputsForLayer(layer) > 1;
end

function iValidateLayerInputName(layerName, layerInputName, validInputNames)
if ~any(strcmp(layerInputName, validInputNames))
    error(message('nnet_cnn:nnet:cnn:LayerGraph:NonExistentLayerInput', layerName, layerInputName));
end
end

function inputNames = iGenerateNumericInputNames(numInputs)
inputNames = arrayfun(@(x)['in' num2str(x)], 1:numInputs, 'UniformOutput', false);
end

function iValidateLayerOutputName(layerName, layerOutputName, validOutputNames)
if ~any(strcmp(layerOutputName, validOutputNames))
    error(message('nnet_cnn:nnet:cnn:LayerGraph:NonExistentLayerOutput', layerName, layerOutputName));
end
end

function iValidateThatConnectionDoesNotExist( ...
    layerOutputInputList, ...
    layerOutputIndex, ...
    layerInputIndex)
if iConnectionExists( ...
    layerOutputInputList, ...
    layerOutputIndex, ...
    layerInputIndex)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:ConnectionAlreadyExists'));
end
end

function tf = iConnectionExists( ...
    layerOutputInputList, ...
    layerOutputIndex, ...
    layerInputIndex)
tf = any(ismember(layerOutputInputList, [layerOutputIndex layerInputIndex], 'rows'));
end

function iValidateConnectionSourceIsCorrectType(source)
if ~iIsValidStringOrCharArray(source)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:SourceMustBeString'));
end
end

function iValidateConnectionDestinationIsCorrectType(destination)
if ~iIsValidStringOrCharArray(destination)
    error(message('nnet_cnn:nnet:cnn:LayerGraph:DestinationMustBeString'));
end
end

function tf = iIsValidStringOrCharArray(x)
tf = nnet.internal.cnn.layer.paramvalidation.isValidStringOrCharArray(x);
end

function iValidateInputForRemoveLayers(layerNames)
if iIsValidStringOrCharArray(layerNames) || iIsValidVectorCellOfStringOrCharArray(layerNames)
else
    error(message('nnet_cnn:nnet:cnn:LayerGraph:InvalidInputForRemoveLayers'));
end
end

function tf = iIsValidVectorCellOfStringOrCharArray(value)
if iscell(value) && isvector(value)
    tf = all(cellfun(@(x)iIsValidStringOrCharArray(x), value));
else
    tf = false;
end
end

function tf = iEdgeExists(edgeIndex)
tf = edgeIndex ~= 0;
end

function iWarnIfThereIsNoReturnArgument(numArguments)
if numArguments == 0
    warning(message('nnet_cnn:nnet:cnn:LayerGraph:NoReturnArgumentLayerGraph'));
end
end

function numInputs = iGetNumberOfInputsForLayer(layer)
internalLayer = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers(layer);
numInputs = internalLayer{1}.numInputs();
end

function numOutputs = iGetNumberOfOutputsForLayer(layer)
internalLayer = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers(layer);
numOutputs = internalLayer{1}.numOutputs();
end