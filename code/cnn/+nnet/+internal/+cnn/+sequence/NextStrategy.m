classdef (Abstract) NextStrategy
    
    properties (Abstract)
        DataSize
        ResponseSize
        PaddingValue
        ReadStrategy
    end

    methods (Abstract)
        % nextBatch
        [miniBatchData, miniBatchResponse, advanceSeq] = nextBatch(this, data, response, indices, seqIndices)
    end
end