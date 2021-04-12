function internalConnections = externalToInternalConnections( externalConnections, layers )
% externalToInternalConnections   Convert external connections to internal
% connections.
%
%   internalConnections = externalToInternalConnections( externalConnections, layers )
%   takes an external connections table and a set of external layers, and
%   converts them to an internal connections matrix.
%
%   Inputs:
%       externalConnections - This format is a table. This is the format
%                             that users will see when they view the
%                             "Connections" property of "DAGNetwork" or
%                             "LayerGraph". The table has two columns:
%                               - Source: A cell array of char arrays.
%                               - Destination: A cell array of char arrays.
%       layers              - An array of external layers
%                             (nnet.cnn.layer.Layer).
%
%   Output:
%       internalConnections - A connections matrix in the "internal"
%                             format. Each row is a connection. The matrix
%                             has four columns, which represent (in order):
%                               - The index of the layer at the start of
%                                 the connection.
%                               - The index of the output for the layer at
%                                 the start of the connection.
%                               - The index of the layer at the end of the
%                                 connection.
%                               - The index of the input for the layer at
%                                 the end of the connection.

%   Copyright 2017 The MathWorks, Inc.

if isempty(externalConnections)
    internalConnections = zeros(0,4);
else
    % externalConnections.Source and externalConnections.Destination are
    % cell arrays of character vectors with elements that may look like
    % this:
    %
    %     {'addition_1/in1'}
    %     {'relu_3'        }
    %     {'conv_4'        }
    %     {'addition_2/in2'}
    %     {'batchnorm_4'   }
    %     {'relu_4'        }
    %
    % The character '/' is the delimiter that separates a layer name from a
    % port name. For example, in 'addition_1/in1', 'addition_1' is the
    % layer name and 'in1' is the port name. We assume that the last
    % character in port name when converted to double represents the port
    % number. These port numbers are the output ports for elements of
    % externalConnections.Source and input ports for elements of
    % externalConnections.Destination. For elements such as 'conv_4' where
    % delimiter '/' is absent, the port number is assumed to be 1.
    %
    % There are a few exceptions to this rule:
    % o For the MaxPooling2DLayer, output ports with names 'out', 'indices'
    % and 'size' map to port numbers 1, 2 and 3 respectively.
    % o For the MaxUnpooling2DLayer, input ports with names 'in', 'indices'
    % and 'size' map to port numbers 1, 2 and 3 respectively.
    
    % Get external layer names.
    externalLayerNames = string({layers.Name}');
    
    % Extract layer names and port names for each connection.
    [sourceLayerNames, sourcePortNames] = ...
        iExtractLayerAndPortNames(externalLayerNames, externalConnections.Source);
    [destinationLayerNames, destinationPortNames] = ...
        iExtractLayerAndPortNames(externalLayerNames, externalConnections.Destination);
    
    % Convert source and destination layer names to layer IDs. If a source
    % or destination layer name matches externalLayerNames{i} then that
    % source or destination layer gets the ID i.
    sourceLayerIDs = iConvertLayerNamesToIDs(sourceLayerNames, externalLayerNames);
    destinationLayerIDs = iConvertLayerNamesToIDs(destinationLayerNames, externalLayerNames);
    
    % Convert source and destination port names to port numbers.
    internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers(layers);
    sourcePortNumbers = iConvertSourcePortNamesToNumbers(sourcePortNames, sourceLayerIDs, internalLayers);
    destinationPortNumbers = iConvertDestinationPortNamesToNumbers(destinationPortNames, destinationLayerIDs, internalLayers);
    
    % Create matrix representing internal connections.
    internalConnections = [sourceLayerIDs, sourcePortNumbers,...
        destinationLayerIDs, destinationPortNumbers];
end
end

function [layerNames,portNames] = iExtractLayerAndPortNames(validLayerNames, connections)
    connections = string(connections);
    validLayerNames = string(validLayerNames);

    % Assume the connection is just the layer name
    layerNames = connections;
    portNames = strings(size(connections));
    
    % Check if there was any invalid layer name and assume that invalid
    % layerNames are names with ports.
    invalid = ~ismember(layerNames, validLayerNames);
    if any(invalid)
        % Parse layerName and portName from the connection
        tokens = regexp(connections(invalid), ...
            "^(?<layer>.*?)(?<port>/[^/]*)?$", 'names', 'forceCellOutput');
        tokens = [tokens{:}];
        
        layerNames(invalid) = [tokens.layer]';
        portNames(invalid) = strrep([tokens.port]', "/", "");
    end
end

function layerIDs = iConvertLayerNamesToIDs(layerNames, externalLayerNames)
[~, layerIDs] = ismember(layerNames, externalLayerNames);
end

function sourcePortNumbers = iConvertSourcePortNamesToNumbers( ...
    sourcePortNames, sourceLayerIDs, internalLayers)
haveSourcePortNames = true;
sourcePortNumbers = iConvertPortNamesToNumbers(sourcePortNames, sourceLayerIDs, internalLayers, haveSourcePortNames);
end

function destinationPortNumbers = iConvertDestinationPortNamesToNumbers( ...
    destinationPortNames, destinationLayerIDs, internalLayers)
haveSourcePortNames = false;
destinationPortNumbers = iConvertPortNamesToNumbers(destinationPortNames, destinationLayerIDs, internalLayers, haveSourcePortNames);
end

function portNumbers = iConvertPortNamesToNumbers(portNames, layerIDs, internalLayers, haveSourcePortNames)
numConnections = numel(portNames);
portNumbers = zeros(numConnections,1);
for i = 1:numConnections
    currentPortName = portNames{i};
    currentLayer = internalLayers{layerIDs(i)};
    if isempty(currentPortName)
        portNumbers(i) = 1;
    else
        portNumbers(i) = iGetPortNumber(currentPortName, currentLayer, haveSourcePortNames);
    end
end
end

function portNumber = iGetPortNumber(portName, layer, haveSourcePortName)
if haveSourcePortName
    portNumber = iGetSourcePortNumber(portName, layer);
else
    portNumber = iGetDestinationPortNumber(portName, layer);
end
end

function sourceIndex = iGetSourcePortNumber(sourceName, layer)
sourceIndex = layer.outputName2Index(sourceName);
end

function destinationIndex = iGetDestinationPortNumber(destinationName, layer)
destinationIndex = layer.inputName2Index(destinationName);
end