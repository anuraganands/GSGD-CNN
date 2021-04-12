classdef (Sealed) ClassificationWithValidationContent < nnet.internal.cnn.util.traininginfo.ContentStrategy
    % ClassificationContent   Classification with validation content strategy
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        FieldNames = {'TrainingLoss', 'TrainingAccuracy', 'ValidationLoss', 'ValidationAccuracy', 'BaseLearnRate'};
        SummaryNames = {'Loss', 'Accuracy', 'ValidationLoss', 'ValidationAccuracy', 'LearnRate'};
    end
end