classdef UserCallbackReporter < nnet.internal.cnn.util.Reporter
    % UserCallbackReporter   Reporter to output training info to user-defined callback
    
    %   Copyright 2016-2017 The MathWorks, Inc.

    properties( Constant )
        InfoFieldsMapToSummaryAndStartValue = { ...
            'Epoch',                'Epoch',                0; ...
            'Iteration',            'Iteration',            0; ...
            'TimeSinceStart',       'Time',                 []; ...
            'TrainingLoss',         'Loss',                 []; ...
            'ValidationLoss',       'ValidationLoss',       []; ...
            'BaseLearnRate',        'LearnRate',            []; ...
            'TrainingAccuracy',     'Accuracy',             []; ...
            'TrainingRMSE',         'RMSE',                 []; ...
            'ValidationAccuracy',   'ValidationAccuracy',   []; ...
            'ValidationRMSE',       'ValidationRMSE',       [] };
    end
    
    properties(Access = private)
        % Callbacks   Functions to call with the info from each iteration
        Callbacks
        
        % Info  Last info value sent to callbacks
        Info
    end
    
    methods
        function this = UserCallbackReporter( callback )
            iAssertIsValidCallback(callback);
            if ~iscell(callback)
                this.Callbacks = { callback };
            else
                this.Callbacks = callback;
            end
            
            % Initialize Info struct with default values
            structArgs = this.InfoFieldsMapToSummaryAndStartValue(:,[1 3])';
            this.Info = struct( structArgs{:} );
        end
        
        function setup( ~ ) 
        end
        
        function start( this )
            % Add state to info then call callbacks
            this.Info.State = "start";            
            this.callCallbacks();
        end
        
        function reportIteration( this, summary )
            % Edit the Info struct with fields from summary input, and call
            % callbacks
            updateInfo( this, summary, "iteration" );
            this.callCallbacks();
        end
        
        function reportEpoch( ~, ~, ~, ~ )
            % End of epoch does not trigger user callback
        end
        
        function finish( this, summary )
            % Incorporate any changes to summary into current info and call
            % callbacks
            updateInfo( this, summary, "done" );
            this.callCallbacks();
        end
    end
    
    methods (Access = private)
        function updateInfo( this, summary, state )
            fields = this.InfoFieldsMapToSummaryAndStartValue(:,1);
            for ii = 1:numel(fields)
                fieldToRecord = fields{ii};
                fieldOfSummary = this.InfoFieldsMapToSummaryAndStartValue{ii, 2};
                this.Info.(fieldToRecord) = iGatherAndConvert( summary.(fieldOfSummary) );
            end
            this.Info.State = state;
        end
        
        function callCallbacks( this )
            % Output to callbacks, retrieving stop output if implemented.
            % Don't call the callbacks directly, call via a wrapper which
            % ensures there is always an output.
            stop = cellfun( @(f) iCallbackWrapper(f, this.Info), this.Callbacks );
            if any( stop )
                notify( this, 'TrainingInterruptEvent' );
            end
        end
    end
end

function iAssertIsValidCallback(callback, ~)
if iscell(callback) && nargin == 1 % Prevents nested cell arrays
    nested = true;
    cellfun(@(c)iAssertIsValidCallback(c, nested), callback);
else
    assert( isa( callback, 'function_handle' ) && nargin(callback) ~= 0 );
end
end

function tf = iCallbackWrapper(f, info)
% Converts any function that takes at least one argument to one that takes
% one and returns one logical output. Also wraps errors with CNNException.
try
    f(info);
    % If function returned a valid output, return it, otherwise return false
    if exist('ans', 'var') == 1 && iIsConvertibleToLogicalScalar(ans) %#ok<NOANS>
        tf = logical(ans); %#ok<NOANS>
    else
        tf = false;
    end
catch me
    err = nnet.internal.cnn.util.CNNException.hBuildUserException( me );
    throw(err);
end
end

function tf = iIsConvertibleToLogicalScalar(x)
tf = isscalar(x) && (isnumeric(x) || islogical(x));
end

function val = iGatherAndConvert(val)
% Gather if gpuArray and convert to double if numeric
if isnumeric(val)
    val = double( gather( val ) );
end
end
