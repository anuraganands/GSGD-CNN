classdef ClassificationMetricRowDataFactory < nnet.internal.cnn.ui.info.MetricRowDataFactory
    % ClassificationMetricRowDataFactory   Implementation of MetricRowDataFactory specifically for classification
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
    end
    
    methods
        function sectionStruct = createMetricRowData(~, infoStruct)
            sectionStruct = struct();
            sectionStruct.RowID = 'VALIDATION_ACCURACY';
            sectionStruct.LeftText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripFinalValidationAccuracyLabel');
            sectionStruct.RightText = iFormattedValidationAccuracy(infoStruct);
        end
    end
    
end

function str = iMessageString(varargin)
m = message(varargin{:});
str = m.getString();
end

function str = iFormattedValidationAccuracy(infoStruct)
validationAccuracy = infoStruct.ValidationAccuracy;
if isempty(validationAccuracy)
   str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripFinalValidationAccuracyNotComputed'); 
else
    numAsStr = num2str(validationAccuracy, '%0.2f');
    str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripFinalValidationAccuracyValue', numAsStr);
end
end
