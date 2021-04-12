classdef (Abstract) RegressionLayer < nnet.internal.cnn.layer.OutputLayer
    % OutputLayer     Interface for convolutional neural network regression
    % output layers
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Abstract)
        % ResponseNames (cellstr)   The names of the responses
        ResponseNames
    end
end
