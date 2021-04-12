function externalConnections = internalToExternalConnections( internalConnections, layers )
% internalToExternalConnections   Convert internal connections to external
% connections
%
%   externalConnections = internalToExternalConnections(internalConnections, layers)
%   takes an internal connections matrix and a set of layers, and converts
%   them to an external layers table.
%
%   Inputs:
%       internalConnections - A matrix in the "internal" fomat. Each row is
%                             a connection. The matrix has four columns, 
%                             which represent (in order):
%                               - The index of the layer at the start of 
%                                 the connection.
%                               - The index of the output for the layer at 
%                                 the start of the connection.
%                               - The index of the layer at the end of the 
%                                 connection.
%                               - The index of the input for the layer at 
%                                 the end of the connection.
%       layers              - An array of external layers 
%                             (nnet.cnn.layer.Layer).
%
%   Output:
%       externalConnections - This format is a table. This is the format
%                             that users will see when they view the 
%                             "Connections" property of "DAGNetwork" or 
%                             "LayerGraph". The table has two columns:
%                               - Source: A cell array of char arrays.
%                               - Destination: A cell array of char arrays.

%   Copyright 2017 The MathWorks, Inc.

if(isempty(internalConnections))
    externalConnections = table( [],{},'VariableNames',{'Source','Destination'} );
else
    sourceList = cellstr({layers(internalConnections(:,1)).Name}.');
    destinationList = cellstr({layers(internalConnections(:,3)).Name}.');
    
    % Get the names for the layer inputs and outputs
    internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers(layers);
    layerOutputList = iGetLayerOutputNames(internalLayers, internalConnections(:,1:2));
    layerInputList = iGetLayerInputNames(internalLayers, internalConnections(:,3:4));
    
    sourceList = iConcatenateCharArrayLists(sourceList, layerOutputList);
    destinationList = iConcatenateCharArrayLists(destinationList, layerInputList);
    
    externalConnections = table( sourceList,destinationList,'VariableNames',{'Source','Destination'} );
end

end

function layerOutputList = iGetLayerOutputNames(layers, sourceConnections)
numConnections = size(sourceConnections,1);
layerOutputList = cell(numConnections,1);
for i = 1:numConnections
    layerIndex = sourceConnections(i,1);
    outputIndexForThisLayer = sourceConnections(i,2);
    layerOutputList{i} = iGetLayerOutputName(layers{layerIndex}, outputIndexForThisLayer);
end
end

function outputName = iGetLayerOutputName(layer, outputIndex)
if layer.numOutputs() == 1 && outputIndex == 1
    outputName = '';
else
    outputName = ['/' layer.outputIndex2Name(outputIndex)];
end
end

function layerInputList = iGetLayerInputNames(internalLayers, destinationConnections)
numConnections = size(destinationConnections,1);
layerInputList = cell(numConnections,1);
for i = 1:numConnections
    layerIndex = destinationConnections(i,1);
    inputIndexForThisLayer = destinationConnections(i,2);
    layerInputList{i} = iGetLayerInputName(internalLayers{layerIndex}, inputIndexForThisLayer);
end
end

function inputName = iGetLayerInputName(layer, inputIndex)
if layer.numInputs() == 1 && inputIndex == 1
    inputName = '';
else
    inputName = ['/' layer.inputIndex2Name(inputIndex)];
end
end

function outputList = iConcatenateCharArrayLists(firstList, secondList)
outputList = strcat(firstList, secondList);
end