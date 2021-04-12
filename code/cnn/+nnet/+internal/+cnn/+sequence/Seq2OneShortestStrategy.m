classdef Seq2OneShortestStrategy < nnet.internal.cnn.sequence.NextStrategy & nnet.internal.cnn.sequence.Utilities

    properties
        DataSize
        ResponseSize
        PaddingValue
        ReadStrategy
    end
    
    methods
        function this = Seq2OneShortestStrategy(readStrategy, dataSize, responseSize, paddingValue)
            this.DataSize = dataSize;
            this.ResponseSize = responseSize;
            this.PaddingValue = paddingValue;
            this.ReadStrategy = readStrategy;
        end
        
        function [miniBatchData, miniBatchResponse, advanceSeq] = nextBatch(this, data, response, indices, ~)
            % nextBatch   Create mini-batches and response where the
            % sequence length is truncated to the length of the shortest
            % sequence in the batch
            
            % advanceSeq is always false. Since we are truncating at
            % the longest sequence length, there is never a need to
            % advance in the sequence dimension
            advanceSeq = false;
            batchSize = numel(indices);
            
            % Get raw data/response at given indices
            rawData = this.ReadStrategy.readDataFcn( data, indices );
            
            % Get data sequence dimensions
            dataSeqLengths = cellfun( @(x)size(x, 2), rawData );
            shortestSeq = min( dataSeqLengths );
            
            % Initialize mini-batch
            miniBatchData = this.initializeMiniBatch( this.DataSize, batchSize, shortestSeq, this.PaddingValue );
            
            % Allocate mini-batch data
            for ii = 1:batchSize
                % Get the data
                dataSeq = rawData{ii};
                % Allocate data into mini-batch
                miniBatchData(:, ii, :) = dataSeq(:, 1:shortestSeq);
            end
            
            % Allocate response
            miniBatchResponse = this.ReadStrategy.readResponseFcn( response, indices );
        end
    end
end