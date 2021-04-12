function hiddenConnections = internalToHiddenConnections(internalConnections)
% internalToHiddenConnections   Convert internal connections to hidden
% connections
%
%   hiddenConnections = internalToHiddenConnections(internalConnections)
%   converts a connections matrix in the "internal" format into a
%   connections table in the "hidden" format.
%
%   Input:
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
%
%   Output:
%       hiddenConnections   - A connections table in the "hidden"
%                             format. The columns are:
%                               - EndNodes: Standard column for the "Edges"
%                                 table of a "digraph" object. A matrix
%                                 where the first column is the index of
%                                 the first layer in the connection, and
%                                 the second column is the index of the
%                                 final layer in the connection.
%                               - EndPorts: A cell array of row
%                                 vectors/matrices. Each entry represents
%                                 the port indices for a given connection.
%                                 An entry will be a matrix if there are
%                                 multiple connections going between the
%                                 same two layers (this is often the case
%                                 with SegNet).


%   Copyright 2017 The MathWorks, Inc.

if isempty(internalConnections)
    hiddenConnections = table(zeros(0,2), cell(0,1), 'VariableNames', {'EndNodes','EndPorts'});
else
    startLayers = internalConnections(:,1);
    stopLayers = internalConnections(:,3);
    endNodes = [startLayers stopLayers];
    endPorts = [internalConnections(:,2) internalConnections(:,4)];
    % At this point, endNodes may have duplicate rows. These duplicate rows
    % indicate multiple connections (with different end ports) between two
    % end nodes. In hiddenConnections below:
    %   EndNodes - captures the unique end nodes
    %   EndPorts - captures all end ports between the end nodes
    [uniqueEndNodes,~,idxUniqueEndNodes] = unique(endNodes,'rows','stable');
    numUniqueEndNodes = size(uniqueEndNodes,1);
    allEndPortsCellArray = arrayfun(@(x) endPorts(idxUniqueEndNodes==x,:),(1:numUniqueEndNodes)','UniformOutput',false);
    hiddenConnections = table(uniqueEndNodes, allEndPortsCellArray, 'VariableNames', {'EndNodes','EndPorts'});
end
end