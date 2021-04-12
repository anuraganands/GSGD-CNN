classdef Recorder < nnet.internal.cnn.util.Reporter
    % Recorder   Reporter to record training info
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % Info (struct)   Struct containing fields to be recorded with
        % training information
        Info
    end
    
    properties(Access = private)
        % Content (nnet.internal.cnn.util.traininginfo.ContentStrategy)
        % Training info content strategy that contains the field names to
        % assign to the output structure and the summary property names to
        % record
        Content
    end
    
    methods
        function this = Recorder( aContentStrategy )
            this.Content = aContentStrategy;
        end
        
        function setup( this )
            this.initializeTrainingInfo();
        end
        
        function start( ~ )
        end
        
        function reportIteration( this, summary )
            % reportIteration   Report an iteration
            this.appendTrainingInfo( summary );
        end
        
        function reportEpoch( ~, ~, ~, ~ )
        end
        
        function finish( this, summary )
            % Incorporate any updates to the summary in the last
            % iteration from other reporters
            this.appendTrainingInfo( summary );
        end
    end
    
    methods (Access = private)
        function initializeTrainingInfo(this)
            % initializeTrainingInfo   Initialize this.Info to an empty
            % structure with the right fields.
            for ii=1:numel(this.Content.FieldNames)
                fieldToRecord = this.Content.FieldNames{ii};
                this.Info.(fieldToRecord) = [];
            end
        end
        
        function appendTrainingInfo(this, summary)
            % appendTrainingInfo   Append the most recent data from the
            % summary to this.Info
            for ii=1:numel(this.Content.FieldNames)
                fieldToRecord = this.Content.FieldNames{ii};
                nameToRecord = this.Content.SummaryNames{ii};
                this.Info.(fieldToRecord)(summary.Iteration) = iGetSummaryProperty(summary, nameToRecord);
            end
        end
    end
end

function val = iGetSummaryProperty(summary, nameToRecord)
% iGetSummaryProperty   Get nameToRecord property value from summary. If it
% is empty, replace it with NaN
val = gather(summary.(nameToRecord));
if isempty(val)
    val = NaN;
end
end
