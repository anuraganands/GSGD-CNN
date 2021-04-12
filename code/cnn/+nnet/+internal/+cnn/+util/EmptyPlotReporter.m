classdef EmptyPlotReporter < nnet.internal.cnn.util.Reporter
    % EmptyPlotReporter   Reporter that is similar to nnet.internal.cnn.util.TrainingPlotReporter, but does nothing. 
    
    % Note that this extends the reporter interface by having an extra
    % function called 'computeFinalValidationResults', similar to the
    % TrainingPlotReporter class.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function setup(~)
        end
        
        function start(~)
        end
        
        function reportIteration(~, ~)
        end
        
        function reportEpoch(~,~,~,~)
        end
        
        function finish(~)
        end
        
        function computeFinalValidationResults(~, ~, ~) 
        end
        
        function finalizePlot(~, ~)
        end
    end
    
end

