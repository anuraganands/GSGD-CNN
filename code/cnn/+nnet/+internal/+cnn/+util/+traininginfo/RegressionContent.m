classdef (Sealed) RegressionContent < nnet.internal.cnn.util.traininginfo.ContentStrategy
    % RegressionContent   Regression content strategy
    
    %   Copyright 2016 The MathWorks, Inc.
    
    properties
        FieldNames = {'TrainingLoss', 'TrainingRMSE', 'BaseLearnRate'};
        SummaryNames = {'Loss', 'RMSE', 'LearnRate'};
    end
end