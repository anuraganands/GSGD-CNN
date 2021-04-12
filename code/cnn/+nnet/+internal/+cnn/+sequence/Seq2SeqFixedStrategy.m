classdef Seq2SeqFixedStrategy < nnet.internal.cnn.sequence.NextStrategy & nnet.internal.cnn.sequence.Utilities
    
    properties
        DataSize
        ResponseSize
        PaddingValue
        ReadStrategy
    end
    
    methods
        function this = Seq2SeqFixedStrategy(readStrategy, dataSize, responseSize, paddingValue)
            this.DataSize = dataSize;
            this.ResponseSize = responseSize;
            this.PaddingValue = paddingValue;
            this.ReadStrategy = readStrategy;
        end
        
        function [miniBatchData, miniBatchResponse, advanceSeq] = nextBatch(this, data, response, indices, seqIndices)
            % nextBatch   Create a mini-batch and response where the
            % sequence length is padded/truncated to a specified length
            
            % Get raw data/response at given indices
            rawData = this.ReadStrategy.readDataFcn( data, indices );
            
            % Initialize mini-batch data
            batchSize = numel( indices );
            sequenceLength = numel( seqIndices );
            miniBatchData = this.initializeMiniBatch( this.DataSize, batchSize, sequenceLength, this.PaddingValue );
            
            % Get sequence dimensions
            dataSeqLengths = cellfun( @(x)size(x, 2), rawData );
            
            % Determine whether another pass along sequence dimension is
            % needed
            advanceSeq = max( dataSeqLengths ) > max( seqIndices );
            
            % Allocate mini-batch data and mini-batch response
            miniBatchResponse = this.initializeMiniBatch( this.ResponseSize, batchSize, sequenceLength, this.PaddingValue );
            for ii = 1:batchSize
                % Get the data
                [dataSeq, dataInds, batchInds] = this.getMiniBatchDataPerObs(rawData, ii, seqIndices, dataSeqLengths(ii));
                % Write into the mini-batch
                miniBatchData(:, ii, batchInds) = dataSeq(:, dataInds);
                % Allocate response
                responseSeq = this.ReadStrategy.readResponseFcn( response, indices(ii) );
                miniBatchResponse(:, ii, batchInds) = responseSeq(:, dataInds);
            end
        end
    end
end