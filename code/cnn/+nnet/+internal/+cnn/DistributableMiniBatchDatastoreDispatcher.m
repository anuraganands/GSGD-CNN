classdef DistributableMiniBatchDatastoreDispatcher < nnet.internal.cnn.DistributableDispatcher
    % DistributableMiniBatchDatastoreDispatcher DistributableDispatcher
    % implementation for MiniBatchDatastore Data Dispatchers.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Constant)
        CanPreserveOrder = true;
    end
    
    methods
        
        function [distributedData, subBatchSizes] = distribute( this, proportions )
            % distribute   Split the dispatcher into partitions according to
            % the given proportions
            
            % Create a cell array of datastores containing one portion of
            % the input datastore per entry in the partitions array.
            [ dsPartitions, subBatchSizes ] = ...
                iSplitDatastore( this.Datastore, this.MiniBatchSize, proportions );
            
            % Create a MiniBatchDatastoreDispatcher containing each of those
            % datastores.
            % Note we always use 'truncateLast' for the endOfEpoch
            % parameters and instead deal with this in the Trainer.
            numPartitions = numel(proportions);
            distributedData = cell(numPartitions, 1);
            for p = 1:numPartitions
                distributedData{p} = nnet.internal.cnn.MiniBatchDatastoreDispatcher( ...
                    dsPartitions{p}, ...
                    subBatchSizes(p), 'truncateLast', this.Precision);
            end
        end
        
    end
end

function [dsPartitions, subBatchSizes] = iSplitDatastore( ...
    data, miniBatchSize, proportions )

import nnet.internal.cnn.DistributableDispatcher.interleavedSelectionByWeight;

numObservations= data.NumObservations;
numPartitions = numel(proportions);
dsPartitions = cell(numPartitions, 1);

% Get the list of indices into the data for each partition
[indicesCellArray, subBatchSizes] = interleavedSelectionByWeight( ...
    numObservations, miniBatchSize, proportions );

% Loop through copying and indexing into the data to create the partitions
for p = 1:numPartitions
    if subBatchSizes(p) > 0
        dsPartitions{p} = data.partitionByIndex(indicesCellArray{p});
    end
end

end


