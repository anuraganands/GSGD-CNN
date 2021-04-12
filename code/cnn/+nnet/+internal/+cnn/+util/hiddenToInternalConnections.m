function internalConnections = hiddenToInternalConnections(hiddenConnections)
% hiddenToInternalConnections   Convert hidden connections to internal
% connections
%
%   internalConnections = hiddenToInternalConnections(hiddenConnections)
%   converts a connections table in the "hidden" format into a connections
%   matrix in the "internal" format.
%
%   Input:
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

if(isempty(hiddenConnections))
    internalConnections = [];
else
    externalEndNodes = hiddenConnections.EndNodes;
    externalEndPorts = hiddenConnections.EndPorts;
    numEndPortsPerEndNodes = cellfun(@(x) size(x,1), externalEndPorts);
    internalEndPorts = cell2mat(externalEndPorts);
    internalEndNodes = [
        repelem(externalEndNodes(:,1),numEndPortsPerEndNodes,1)...
        repelem(externalEndNodes(:,2),numEndPortsPerEndNodes,1)];
    internalConnections = [
        internalEndNodes(:,1), ...
        internalEndPorts(:,1), ...
        internalEndNodes(:,2), ...
        internalEndPorts(:,2)];
    
    % Sort internal connections into order by:
    %  - 1st source layer
    %  - 2nd source layer output
    %  - 3rd destination layer
    %  - 4th destination layer output
    internalConnections = sortrows(internalConnections, [1 2 3 4]);
end
end