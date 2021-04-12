classdef DistributableDispatcher < handle
    % DistributableDispatcher Mixin for a DataDispatcher to give it the
    % capability of being wrapped by a DistributedDispatcher.
    
    %   Copyright 2016 The MathWorks, Inc.
    
    properties (Constant, Abstract)
        CanPreserveOrder
    end
    
    methods (Abstract)
        
        [distributedData, subBatchSizes] = distribute( this, proportions );
        
    end
    
    % Some helper utilities for distribution
    methods (Static)
        
        function portions = partitionByWeight(amount, weights)
        % Divide a quantity according to some weights ensuring integer
        % amounts in each portion.
            portions = floor( amount * weights(:)' ); % Always row vector
            
            % Adjust to ensure the overall size is unchanged, by dealing out
            % remaining data evenly while ignoring zero-weighted portions
            remainder = amount - sum( portions );
            inc = ones( 1, remainder );
            mask = weights > 0;
            numPortions = sum( mask );
            padLength = numPortions - mod( remainder, numPortions );
            inc = reshape( [ inc, zeros( 1, padLength ) ], [], numPortions );
            portions(mask) = portions(mask) + sum( inc, 1 );
        end
        
        function [indicesCellArray, portions] = interleavedSelectionByWeight(numObservations, batchSize, weights)
        % Create a cell array of lists of indices into some observations
        % divided into batches, with each batch containing an appropriate
        % proportion of the observations. This code is used by most
        % dispatcher types to divide the data up in the right proportions
        % and yet create the exact same batches across all workers that
        % would be created in a serial version. To ensure even
        % representation across response categories and groupings the data
        % should be shuffled before being partitioned by the output from
        % this function.
        %
        % For example, if the data is
        %    [1 2 3 4 5 6 7 8 9 10 11 12 13 14]
        % the batch size is 8 and we are partitioning into two with weights
        % of [0.75 0.25], the output indicesCellArray will be
        %    {1} = [1 2 3 4 5 6 | 9 10 11 12 13]
        %    {2} = [7 8 | 14]
        % The final batch divides the remaining data appropriately and so
        % each partition will usually get less of the data than in the
        % other batches.
        
            import nnet.internal.cnn.DistributableDispatcher.partitionByWeight;
            portions = partitionByWeight(batchSize, weights);
            numWholeBatches = floor(numObservations / batchSize);
            lastBatchSize = mod(numObservations, batchSize);
    
            % Shape a list of indices to have one minibatch per column
            numObservationsInWholeBatches = numWholeBatches * batchSize;
            indices = reshape(1:numObservationsInWholeBatches, batchSize, numWholeBatches);
            
            % Determine the indices in the final minibatch
            if lastBatchSize > 0
                lastBatchIndices = (numObservationsInWholeBatches+1):numObservations;
                lastMiniBatchSubBatchSizes = partitionByWeight(lastBatchSize, weights);
                lastBatchIndicesPerWorker = mat2cell( lastBatchIndices', lastMiniBatchSubBatchSizes );
            end
            
            % Loop through cropping the indices into the data for each
            % partition. The way we have arranged the index array this
            % means dividing it up by rows.
            numPartitions = numel(weights);
            indicesCellArray = cell(numPartitions, 1);
            startRowPerPartition = [1 cumsum(portions)+1];
            for p = 1:numPartitions
                if portions(p) > 0
                    startRow = startRowPerPartition(p);
                    endRow = startRow + portions(p) - 1;
                    indicesCellArray{p} = indices(startRow:endRow, :);
                    % Add the indices for the final minibatch
                    if lastBatchSize > 0
                        indicesCellArray{p} = [indicesCellArray{p}(:); lastBatchIndicesPerWorker{p}];
                    end
                    indicesCellArray{p} = indicesCellArray{p}(:);
                end
            end
        end
        
    end

end
