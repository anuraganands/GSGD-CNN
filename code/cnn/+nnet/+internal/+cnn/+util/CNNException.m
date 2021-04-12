%CNNException Capture error information from errors thrown by CNNs and provide more informative reporting.
%
%   CNNException is an MException and inherits its methods and properties.
%
%   Additional or modified CNNException properties:
%      UnderlyingCause - The MException that caused this CNNException
%      cause           - MException representing the UnderlyingCause with a
%                        modified message for reporting
%
%   See also MException, ParallelException

%   Copyright 2017 The MathWorks, Inc.

classdef CNNException < MException
% The purpose of CNNException is to improve reporting of errors occurring
% when training or running neural networks. It hides the internal call
% stack while still reporting the line of code that errored, and the
% complete stack of user code if the error occurred in user code. It also
% retains the original exception so that a complete trace is still
% available to tech support or development.
%
% User code should be wrapped in a call to
% nnet.internal.cnn.util.fevalUserCode, which creates a CNNException to
% capture the error report, keeping only the user code portion of that
% report.
%
% At the entry point to CNN code errors can be caught and wrapped in a
% CNNException, which when thrown will create the improved report, while
% retaining all the underlying true information.

    properties( SetAccess = private )
        % UnderlyingCause   Contains the original exception and can be
        %                   recovered on the command line using
        %                   MException.last.UnderlyingCause.
        UnderlyingCause
    end
    
    properties( Hidden, SetAccess = private )
        UserCodeReport
    end
    
    methods( Static, Hidden )
        
        function exception = hBuildCustomError( cause )
        % hBuildCustomError  Build a CNNException for display, with some
        % special manipulation of the information and showing only user
        % code in the stack
            % Default identifier and message will come from the cause
            id = [];
            msg = [];

            % Exceptions with 'internal' in the ID have that stripped
            if contains(cause.identifier, ':internal')
                id = strrep(cause.identifier, ':internal', '');
            end
            
            % If the cause is a ParallelException the message should not be
            % the ParallelException message ("Error on worker 1" or
            % what have you) it should be the message from the remotecause
            remotecause = cause;
            if isa( cause, 'ParallelException' )
                remotecause = iUnpeelParallelException(cause);
                msg = remotecause.message;
            end

            % Datastore's custom ReadFcn error sets the error message to
            % the whole report. We want this for the cause, but not also
            % for the main message, so reset it to the original message,
            % stored with the exception as a HiddenCause property.
            if remotecause.identifier == "MATLAB:datastoreio:customreaddatastore:readFcnError"
                msg = remotecause.HiddenCause.message;
            end

            % GPU Out-Of-Memory gets given a more helpful message
            if remotecause.identifier == "parallel:gpu:array:OOM"
                id = 'nnet_cnn:trainNetwork:GPUOutOfMemory';
                msg = getString( message( id ) );
            end

            % Create the exception
            exception = nnet.internal.cnn.util.CNNException(id, msg, cause);

            % If (remote)cause is a CNNException, change the message of the
            % cause so that it displays the whole stack. We use the
            % remotecause here so that ParallelExceptions will show as if
            % serial if they occurred in user code.
            if isa( remotecause, 'nnet.internal.cnn.util.CNNException' )
                cause = MException( remotecause.identifier, '%s', ...
                    remotecause.UserCodeReport );
            end
            
            % Ensure that the error report includes at least the message
            % and top of the stack of the cause
            if remotecause.identifier == "nnet_cnn:internal:cnn:analyzer:NetworkAnalyzer:NetworkHasErrors"
                for i=1:numel(remotecause.cause)
                    exception = addCause( exception, remotecause.cause{i} );
                end
                
                if iIsLiveScript()
                    exception = nnet.internal.cnn.util.CNNException(id, exception.getReport(), cause);
                end
            else
                exception = addCause( exception, cause );
            end
        end
        
        function exception = hBuildUserException( cause )
        % hBuildUserException  Build a CNNException to capture an error
        % occurring in user code and record the stack for the user code
        % only. This is expected to be thrown and caught again by the entry
        % point code.
            currentStack = iGetCurrentStack();
            exception = nnet.internal.cnn.util.CNNException([], [], cause);
            exception.UserCodeReport = iCropReportBelowFile( getReport(cause), currentStack(2).file );
        end
    end
    
    methods( Access = private )
        
        function this = CNNException( id, msg, cause )
        % CNNException  Constructor. Essentially an MException with
        % additional UnderlyingCause.
            if isempty(id)
                id = cause.identifier;
            end
            if isempty(msg)
                msg = cause.message;
            end
            this = this@MException(id, '%s', msg);
            this.UnderlyingCause = cause;
        end
        
    end
    
end

function report = iCropReportBelowFile(report, file)
    % Find the start of the stack for the file
    idx = strfind(report, file);
    % Take the last new line index
    idx = regexp(report(1:idx(1)), '\n');
    % Remove all the unwanted stack
    report = strtrim( report(1:idx(end)) );
end

function exception = iUnpeelParallelException( exception )
% Return the underlying remote cause inside a ParallelException
if isa( exception, 'ParallelException' )
    % If multiple workers threw different errors it's possible we could try
    % to display them all, but this is unusual and/or inadvisable for CNNs,
    % which use synchronous parallel constructs. Pick the first error.
    % If there is no remotecause it's usually because the worker
    % unexpectedly died or communication was lost. In this case leave the
    % exception as the original parallel exception.
    if ~isempty(exception.remotecause)
        exception = iUnpeelParallelException( exception.remotecause{1} );
    end
end
end

function stack = iGetCurrentStack()
try
    error('nnet:internal:cnn:DummyError', 'This is a dummy error')
catch err
    stack = err.stack;
    % Remove this function from the stack
    stack = stack(2:end);
end
end

function tf = iIsLiveScript()
    stack = dbstack();
    tf = any({stack.name} == "EvaluationOutputsService.evalRegions");
end
