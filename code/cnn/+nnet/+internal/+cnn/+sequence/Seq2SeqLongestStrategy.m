classdef Seq2SeqLongestStrategy < nnet.internal.cnn.sequence.NextStrategy & nnet.internal.cnn.sequence.Utilities

    properties
        DataSize
        ResponseSize
        PaddingValue
        ReadStrategy
    end
    
    methods
        function this = Seq2SeqLongestStrategy(readStrategy, dataSize, responseSize, paddingValue)
            this.DataSize = dataSize;
            this.ResponseSize = responseSize;
            this.PaddingValue = paddingValue;
            this.ReadStrategy = readStrategy;
        end
        
        function [miniBatchData, miniBatchResponse, advanceSeq] = nextBatch(this, data, response, indices, ~)
            % nextBatch   Create mini-batches and response where the
            % sequence length is padded to the length of the longest
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
            longestSeq = max( dataSeqLengths );
            
            % Initialize mini-batch
            miniBatchData = this.initializeMiniBatch( this.DataSize, batchSize, longestSeq, this.PaddingValue );
            
            % Allocate mini-batch data and mini-batch response
            miniBatchResponse = this.initializeMiniBatch( this.ResponseSize, batchSize, longestSeq, this.PaddingValue );
            for ii = 1:batchSize
                % Get the data
                dataSeq = rawData{ii};
                % Allocate data into mini-batch
                miniBatchData(:, ii, 1:dataSeqLengths(ii)) = dataSeq;
                % Allocate response
                responseSeq = this.ReadStrategy.readResponseFcn( response, indices(ii) );
                miniBatchResponse(:, ii, 1:dataSeqLengths(ii)) = responseSeq;
            end
        end
    end
end