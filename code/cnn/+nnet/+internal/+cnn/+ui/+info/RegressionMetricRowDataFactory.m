classdef RegressionMetricRowDataFactory < nnet.internal.cnn.ui.info.MetricRowDataFactory
    % RegressionMetricRowDataFactory   Implementation of MetricRowDataFactory specifically for regression
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
    end
    
    methods
        function sectionStruct = createMetricRowData(~, infoStruct)
            sectionStruct = struct();
            sectionStruct.RowID = 'VALIDATION_RMSE';
            sectionStruct.LeftText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripFinalValidationRMSELabel');
            sectionStruct.RightText = iFormattedValidationRMSE(infoStruct);
        end
    end
    
end

function str = iMessageString(varargin)
m = message(varargin{:});
str = m.getString();
end

function str = iFormattedValidationRMSE(infoStruct)
validationRMSE = infoStruct.ValidationRMSE;
if isempty(validationRMSE)
   str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripFinalValidationRMSENotComputed'); 
else
    numSigFig = 5;
    numAsStr = num2str(validationRMSE, numSigFig);
    str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripFinalValidationRMSEValue', numAsStr);
end
end
