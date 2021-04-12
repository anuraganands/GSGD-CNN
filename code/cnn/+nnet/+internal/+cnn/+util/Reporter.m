classdef(Abstract) Reporter < handle
    % Reporter   Convolutional neural networks training reporter interface
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    events
        % Fires to attempt to interrupt training
        TrainingInterruptEvent
    end
    
    methods(Abstract)
        % setup   Setup the reporter. This should be called before any
        % training starts.
        setup( this )
        
        % start   Start the reporter. This should only be called in a
        % Trainer.train call.
        start( this )
        
        % reportIteration   Report an iteration
        reportIteration( this, summary )
        
        % reportEpoch   Report an epoch
        reportEpoch( this, epoch, iteration, network )
        
        % finish   End the reporter
        finish( this, summary )
    end
    
    methods
        function computeIteration( this, summary, network ) %#ok<INUSD>
            % computeIteration   Perform computations with network and
            % summary for a certain iteration
            
            % Do nothing by default
        end
        
        function computeFinish( this, summary, network ) %#ok<INUSD>
            % computeFinish   Perform wrap-up computations
            
            % Do nothing by default
        end
    end
end
