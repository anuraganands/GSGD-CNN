classdef SummaryFactory < handle
    % SummaryFactory   Creates a summary
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function summary = createSummary(~)
            summary = nnet.internal.cnn.util.MiniBatchSummary(); 
        end
    end
    
end

