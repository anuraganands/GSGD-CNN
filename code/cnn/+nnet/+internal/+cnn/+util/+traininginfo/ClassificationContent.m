classdef (Sealed) ClassificationContent < nnet.internal.cnn.util.traininginfo.ContentStrategy
    % ClassificationContent   Classification content strategy
    
    %   Copyright 2016 The MathWorks, Inc.
    
    properties
        FieldNames = {'TrainingLoss', 'TrainingAccuracy', 'BaseLearnRate'};
        SummaryNames = {'Loss', 'Accuracy', 'LearnRate'};
    end
end