classdef VectorReporter < nnet.internal.cnn.util.Reporter
    % VectorReporter   Container to hold a series of reporters
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties
        Reporters
    end
    
    methods
        function setup( this )
            computeAndReport( this, 'setup' );
        end
        
        function start( this )
            computeAndReport( this, 'start' );
        end
        
        function reportIteration( this, summary )
            computeAndReport( this, 'reportIteration', summary );
        end
        
        function computeIteration( this, summary, network )
            computeAndReport( this, 'computeIteration', summary, network );
        end
        
        function reportEpoch( this, epoch, iteration, network )
            computeAndReport( this, 'reportEpoch', epoch, iteration, network );
        end
        
        function finish( this, summary )
            computeAndReport( this, 'finish', summary );
        end
        
        function computeFinish( this, summary, net )
            computeAndReport( this, 'computeFinish', summary, net );
        end
        
        function add( this, reporter )
            % Merge vector reporters
            if isa( reporter, 'nnet.internal.cnn.util.VectorReporter' )
                this.Reporters = cat(1, this.Reporters, reporter.Reporters);
            else
                this.Reporters = cat(1, this.Reporters, { reporter });
            end
            
            % A vector reporter must forward events fired by its members
            addlistener( reporter, 'TrainingInterruptEvent', ...
                @(~,~)notify(this, 'TrainingInterruptEvent') );
        end
    end
    
    methods( Access = private )
        function computeAndReport( this, method, varargin )
            for i = 1:length(this.Reporters)
                feval( method, this.Reporters{i}, varargin{:} );
            end
        end
    end
end
