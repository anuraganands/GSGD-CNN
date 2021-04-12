classdef EmptyValidationReporter < nnet.internal.cnn.util.Reporter
    % EmptyValidationReporter   A reporter that does nothing except setting
    % the validation results to empty in computeIteration
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function setup(~)
        end
        
        function start(~)
        end
        
        function reportIteration(~,~)
        end
        
        function reportEpoch(~,~,~,~)
        end
        
        function finish(~)
        end
        
        function computeIteration(~, summary, ~)
            summary.ValidationLoss = [];
            summary.ValidationPredictions = [];
            summary.ValidationResponse = [];
        end
        
        function computedSummary = computeFinalValidationResultForPlot(this, summary, net) %#ok<INUSL,INUSD>
            computedSummary = summary;
            computedSummary.ValidationLoss = [];
            computedSummary.ValidationPredictions = [];
            computedSummary.ValidationResponse = [];
        end
    end
    
end

