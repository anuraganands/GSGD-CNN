classdef StopReasonRowDataFactory < handle
    % StopReasonRowDataFactory   Creates table data for indicating why training stopped
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function sectionStruct = createStopReasonRowData(~, stopReason)
            sectionStruct = struct();
            sectionStruct.RowID = 'STOP_REASON';
            sectionStruct.LeftText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripStopReasonLabel');
            sectionStruct.RightText = iFormattedStopReason(stopReason);
        end
    end
    
end

% helpers
function str = iMessageString(varargin)
m = message(varargin{:});
str = m.getString();
end

function str = iFormattedStopReason(stopReason)
switch stopReason
    case 'FinalIteration'
        str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripStopReasonValueFinalIter');
    case 'StopButton'
        str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripStopReasonValueStopButton');
    case 'ValidationStopping'
        str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripStopReasonValueValidationStopping');
    case 'OutputFcn'
        str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripStopReasonValueOutputFcn');
    otherwise
        warning(message('nnet_cnn:internal:cnn:ui:trainingplot:GenericWarning'));
        str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripStopReasonValueFinalIter');
end
end
