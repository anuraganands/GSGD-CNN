classdef DoesNotThrowErrors < matlab.unittest.constraints.Constraint
    % DoesNotThrowErrors   Constraint to verify that a function does not
    % throw any errors
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties(SetAccess=immutable)
        FunctionName = ''
        Nargout = 0
    end
    
    methods
        
        function constraint = DoesNotThrowErrors(functionName,numArgout)
            constraint.FunctionName = functionName;
            if nargin == 2
                constraint.Nargout = numArgout;
            end
        end
        
        function [bool, errorMsg] = satisfiedBy(constraint, actual)
            bool = true;
            errorMsg = [];
            try
                [out{1:constraint.Nargout}] = actual(); %#ok<NASGU>
            catch e
                % An error was thrown
                bool = false;
                errorMsg = iGetReport(e);
            end
        end
        
        function diag = getDiagnosticFor(constraint, actual)
            import matlab.unittest.diagnostics.StringDiagnostic
            
            [bool, errorMsg] = constraint.satisfiedBy(actual);
            if bool
                diag = '';
            else
                % An error was thrown
                diag = StringDiagnostic( ...
                    getString( message( 'nnet_cnn:nnet:checklayer:constraints:DoesNotThrowErrors:ErrorThrown', ...
                    constraint.FunctionName, errorMsg ) ) );
            end
        end
    end
end

function errorMsg = iGetReport(exception)
% iGetReport   Get the error message and stack trace for the exception
% thrown by the function under test. The stack trace should only include
% functions of the custom layer that is being tested by checkLayer.
cropKeyword = 'nnet.checklayer.';
errorMsg = iCropReportAtKeyword( exception.getReport(), cropKeyword );
end

function croppedReport = iCropReportAtKeyword(report,cropWord)
    % Parse stack trace and crop report at the first occurrence of the cropWord    
    cropIdx = strfind(report,cropWord);
    newLineIdx = regexp(report, '\n\n');    
    % Find last empty line before cropWord
    cutoffIdx = newLineIdx( find(newLineIdx < cropIdx(1), 1, 'last') );
    croppedReport = report(1:cutoffIdx-1);
end