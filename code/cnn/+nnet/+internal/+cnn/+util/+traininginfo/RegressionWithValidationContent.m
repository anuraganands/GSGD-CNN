classdef (Sealed) RegressionWithValidationContent < nnet.internal.cnn.util.traininginfo.ContentStrategy
    % RegressionWithValidationContent   Regression with validation content strategy
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        FieldNames = {'TrainingLoss', 'TrainingRMSE', 'ValidationLoss', 'ValidationRMSE', 'BaseLearnRate'};
        SummaryNames = {'Loss', 'RMSE', 'ValidationLoss', 'ValidationRMSE', 'LearnRate'};
    end
end