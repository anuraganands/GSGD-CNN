classdef Utilities
    
    methods (Static)
        function miniBatchData = initializeMiniBatch(dataSize, batchSize, sequenceLength, paddingValue )
            % initializeMiniData   Create an array which will have the
            % mini-batch data assigned into it
            miniBatchData = paddingValue.*ones( dataSize, batchSize, sequenceLength );
        end
        
        function [data, dataInds, batchInds] = getMiniBatchDataPerObs(data, index, seqIndices, dataSeqLength)
            % Get the data
            data = data{ index };
            % Get indices for indexing into the data
            dataInds = seqIndices(1):min(seqIndices(end), dataSeqLength);
            % Get indices for indexing into the mini-batch
            batchInds = 1:numel(dataInds);
        end
    end
end