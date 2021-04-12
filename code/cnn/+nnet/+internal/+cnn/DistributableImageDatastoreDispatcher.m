classdef DistributableImageDatastoreDispatcher < nnet.internal.cnn.DistributableDispatcher
    % DistributableImageDatastoreDispatcher DistributableDispatcher
    % implementation for ImageDatastore Data Dispatchers.
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
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
            
            % Create an ImageDatastoreDispatcher containing each of those
            % datastores.
            % Note we always use 'truncateLast' for the endOfEpoch
            % parameters and instead deal with this in the Trainer.
            numPartitions = numel(proportions);
            distributedData = cell(numPartitions, 1);
            for p = 1:numPartitions
                distributedData{p} = nnet.internal.cnn.ImageDatastoreDispatcher( ...
                    dsPartitions{p}, ...
                    subBatchSizes(p), 'truncateLast', this.Precision);
            end
        end
        
    end
end

function [dsPartitions, subBatchSizes] = iSplitDatastore( ...
    data, miniBatchSize, proportions )
% Divide up table by rows according to the given proportions

import nnet.internal.cnn.DistributableDispatcher.interleavedSelectionByWeight;

numObservations= numel(data.Files);
numPartitions = numel(proportions);
dsPartitions = cell(numPartitions, 1);
correctCategories = categories(data.Labels);

% Get the list of indices into the data for each partition
[indicesCellArray, subBatchSizes] = interleavedSelectionByWeight( ...
    numObservations, miniBatchSize, proportions );

% Loop through copying and indexing into the data to create the partitions
for p = 1:numPartitions
    if subBatchSizes(p) > 0
        % Take a copy of the datastore
        subds = copy(data);
        
        % Prune all files not in the partition. This correctly selects
        % Labels as well.
        mask = true(numObservations, 1);
        mask(indicesCellArray{p}) = false;
        subds.Files(mask) = [];
        
        % Make sure all the categories in the partitions are the same as
        % for the original
        iSetCategories(subds, correctCategories);
        
        % Set in output
        dsPartitions{p} = subds;
    end
end
end

function iSetCategories( ds, cats )
if ~isempty(ds.Files)
    ds.Labels = setcats( ds.Labels, cats );
end
end
