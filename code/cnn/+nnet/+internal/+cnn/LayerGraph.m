classdef LayerGraph
    % LayerGraph   An internal LayerGraph
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % Layers   The layers for the LayerGraph
        %   A cell array of internal layers.
        Layers
        
        % Connections   The connections for the LayerGraph
        %   A matrix specifying connections between layers. Each row is a
        %   connection. The matrix has four columns, which represent (in
        %   order):
        %     - The index of the layer at the start of the connection.
        %     - The index of the output for the layer at the start of the
        %       connection.
        %     - The index of the layer at the end of the connection
        %     - The index of the input for the layer at the end of the.
        %       connection.
        Connections
    end
    
    properties(Dependent, Access=private)
        % Digraph   A digraph object that represents the LayerGraph
        %   A digraph object representing the LayerGraph. This is useful
        %   for running graph related algorithms.
        Digraph
        
        % AugmentedDigraph
        AugmentedDigraph
    end
    
    methods
        function val = get.Digraph(this)
            val = this.createDigraph();
        end
        
        function val = get.AugmentedDigraph(this)
            val = this.createAugmentedDigraph();
        end
    end
    
    methods
        function this = LayerGraph(layers, connections)
            % LayerGraph   An internal LayerGraph object
            %
            %   this = LayerGraph(layers, connections) takes the following
            %   inputs:
            %       - A cell array of internal layers. Note that the layers
            %         should have unique names.
            %       - A matrix of connections. See the Connections property
            %         for a description of the format.
            
            this.Layers = layers;
            this.Connections = connections;
        end
        
        function plot(this)
            % plot   Plot the LayerGraph
            
            nodeNames = this.Digraph.Nodes.Name;
            this.Digraph.plot('NodeLabel',nodeNames,'Layout','layered');
        end
        
        function sortedIndices = topologicalSort(this)
            % topologicalSort   Return a list of layer indices in topological order
            %
            %   sortedIndices = topologicalSort(this) returns a row vector
            %   of layer indices in topological order.
            
            sortedIndices = this.Digraph.toposort('Order','stable');
        end
        
        function [sortedIndices, sortedGraph] = topologicalSortTemp(this)
            
            [sortedIndices, sortedGraph] = this.AugmentedDigraph.toposort('Order','stable');
        end
        
        function augmentedDigraph = getAugmentedDigraph(this)
            augmentedDigraph = this.AugmentedDigraph;
        end
        
        function [this,topologicalOrder] = toposort(this)
            topologicalOrder  = this.AugmentedDigraph.toposort('Order','stable');
            this.Layers = nnet.internal.cnn.LayerGraph.originalToSortedLayers(this.Layers, topologicalOrder);
            this.Connections = nnet.internal.cnn.LayerGraph.originalToSortedConnections(this.Connections, topologicalOrder);
        end
    end
    
    methods(Access = private)
        function graph = createDigraph(this)
            layerNames = iGetInternalLayerNames(this.Layers);
            edgeTable = iCreateEdgeTable(this.Connections);
            nodeTable = iCreateNodeTable(layerNames);
            graph = digraph( ...
                edgeTable, ...
                nodeTable);
        end
        
        function graph = createAugmentedDigraph(this)
            layerNames = iGetInternalLayerNames(this.Layers);
            edgeTable = iCreateAugmentedEdgeTable(this.Connections);
            nodeTable = iCreateNodeTable(layerNames);
            graph = digraph( ...
                edgeTable, ...
                nodeTable);
        end
    end
    
    methods(Static,Hidden)
        % Suppose layers is the unsorted layer array and sortedLayers is
        % the topologically sorted layer array. Then topologicalOrder is a
        % vector of indices such that the following condition holds:
        %
        %   sortedLayers = layers(topologicalOrder)
        %
        % In other words, index topologicalOrder(k) for layers corresponds
        % to index k for sortedLayers.
        
        function transformedIndices = originalToSortedLayerIndices(indices, topologicalOrder)
            % This function accepts indices for layers array and converts
            % them into indices for the sortedLayers array.
            indices = indices(:);
            topologicalOrder = topologicalOrder(:);
            sortedLayerIndices = (1:numel(topologicalOrder))';
            [~,loc] = ismember(indices,topologicalOrder);
            transformedIndices = sortedLayerIndices(loc);
        end
        
        function transformedIndices = sortedToOriginalLayerIndices(indices, topologicalOrder)
            % This function accepts indices for sortedLayers array and
            % converts them into indices for the layers array.
            indices = indices(:);
            topologicalOrder = topologicalOrder(:);
            sortedLayerIndices = (1:numel(topologicalOrder))';
            [~,loc] = ismember(indices,sortedLayerIndices);
            transformedIndices = topologicalOrder(loc);
        end
        
        function sortedLayers = originalToSortedLayers(layers, topologicalOrder)
            sortedLayers = layers(topologicalOrder);
        end
        
        function sortedConnections = originalToSortedConnections(connections, topologicalOrder)
            sourceLayerIDs = connections(:,1);
            targetLayerIDs = connections(:,3);
            sortedSourceLayerIDs = nnet.internal.cnn.LayerGraph.originalToSortedLayerIndices(sourceLayerIDs, topologicalOrder);
            sortedTargetLayerIDs = nnet.internal.cnn.LayerGraph.originalToSortedLayerIndices(targetLayerIDs, topologicalOrder);
            sortedConnections = [sortedSourceLayerIDs(:), connections(:,2), sortedTargetLayerIDs(:), connections(:,4)];
        end
        
        function layers = sortedToOriginalLayers(sortedLayers, topologicalOrder)
            originalIndices = (1:numel(sortedLayers))';
            sortedIndices = nnet.internal.cnn.LayerGraph.originalToSortedLayerIndices(originalIndices, topologicalOrder);
            layers = sortedLayers(sortedIndices);
        end
        
        function connections = sortedToOriginalConnections(sortedConnections, topologicalOrder)
            sortedSourceLayerIDs = sortedConnections(:,1);
            sortedTargetLayerIDs = sortedConnections(:,3);
            sourceLayerIDs = nnet.internal.cnn.LayerGraph.sortedToOriginalLayerIndices(sortedSourceLayerIDs, topologicalOrder);
            targetLayerIDs = nnet.internal.cnn.LayerGraph.sortedToOriginalLayerIndices(sortedTargetLayerIDs, topologicalOrder);
            connections = [sourceLayerIDs(:), sortedConnections(:,2), ...
                targetLayerIDs(:), sortedConnections(:,4)];
        end
    end
end

function names = iGetInternalLayerNames(layers)
names = cellfun(@(layer)layer.Name, layers, 'UniformOutput', false);
end

function edgeTable = iCreateEdgeTable(connectionsMatrix)
startLayers = connectionsMatrix(:,1);
stopLayers = connectionsMatrix(:,3);
endNodes = [startLayers stopLayers];
endNodes = unique(endNodes,'rows','stable');
edgeTable = table(endNodes, 'VariableNames', {'EndNodes'});
end

function edgeTable = iCreateAugmentedEdgeTable(connectionsMatrix)
startLayers = connectionsMatrix(:,1);
stopLayers = connectionsMatrix(:,3);
endNodes = [startLayers stopLayers];
endPorts = [connectionsMatrix(:,2) connectionsMatrix(:,4)];
% At this point, endNodes may have duplicate rows. These duplicate rows
% indicate multiple connections (with different end ports) between two end
% nodes. In edgeTable below:
%   EndNodes    - captures the unique end nodes
%   EndPorts    - captures one of the end ports between the end nodes
%   AllEndPorts - captures all end ports between the end nodes
[uniqueEndNodes,idxEndNodes,idxUniqueEndNodes] = unique(endNodes,'rows','stable');
uniqueEndPorts = endPorts(idxEndNodes,:);
uniqueEndPortsCellArray = num2cell(uniqueEndPorts, 2);
numUniqueEndNodes = size(uniqueEndNodes,1);
allEndPortsCellArray = arrayfun(@(x) endPorts(idxUniqueEndNodes==x,:),(1:numUniqueEndNodes)','UniformOutput',false);
edgeTable = table(uniqueEndNodes, uniqueEndPortsCellArray, allEndPortsCellArray, 'VariableNames', {'EndNodes','EndPorts','AllEndPorts'});
end

function nodeTable = iCreateNodeTable(layerNames)
nodeTable = table(layerNames, 'VariableNames', {'Name'});
end