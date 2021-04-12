classdef DistributableFourDArrayDispatcher < nnet.internal.cnn.DistributableDispatcher
    % DistributableFourDArrayDispatcher DistributableDispatcher
    % implementation for FourDArray Data Dispatchers.
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties (Constant)
        CanPreserveOrder = true;
    end
    
    methods
        
        function [distributedData, subBatchSizes] = distribute( this, proportions )
        % distribute   Split the dispatcher into partitions according to
        % the given proportions
            
            % Create a cell containing one partition of the data per
            % entry in the proportions array
            [ partitionedData, partitionedResponse, subBatchSizes ] = ...
                iSplitData( this.Data, this.Response, this.MiniBatchSize, ...
                this.OrderedIndices, proportions );
            
            % Create a new FourDArrayDispatcher wrapping each of those
            % tables.
            % Note we always use 'truncateLast' for the endOfEpoch
            % parameters and instead deal with this in the Trainer.
            numPartitions = numel(proportions);
            distributedData = cell(numPartitions, 1);
            for p = 1:numPartitions
                distributedData{p} = nnet.internal.cnn.FourDArrayDispatcher( ...
                    partitionedData{p}, partitionedResponse{p}, ...
                    subBatchSizes(p), 'truncateLast', this.Precision);
            end
        end

    end
end

function [ dataPartitions, responsePartitions, subBatchSizes ] = iSplitData( ...
    data, response, miniBatchSize, currentOrder, proportions )
% Divide up the data and response according to the given proportions

import nnet.internal.cnn.DistributableDispatcher.interleavedSelectionByWeight;

numObservations= size(data,4);
numPartitions = numel(proportions);
dataPartitions = cell(numPartitions, 1);
responsePartitions = cell(numPartitions, 1);

% Validate that the sizes match
if ismatrix(response)
    responseObsDim = 1;
else
    responseObsDim = 4;
end
assert(size(response,responseObsDim) == size(data,4), 'Response size does not match data');

% Get the list of indices into the data for each partition
[indicesCellArray, subBatchSizes] = interleavedSelectionByWeight( ...
    numObservations, miniBatchSize, proportions );

% Loop through indexing into the data to create the partitions
for p = 1:numPartitions
    if subBatchSizes(p) > 0
        % Index data to create partition
        indices = currentOrder(indicesCellArray{p});
        dataPartitions{p} = data(:, :, :, indices);
        % If response is a vector or 2D array, it should be distributed
        % along its rows. If it's a 4D array, it should be distributed
        % along dim 4.
        if ismatrix(response)
            responsePartitions{p} = response(indices, :);
        else
            responsePartitions{p} = response(:, :, :, indices);
        end
    end
end
end