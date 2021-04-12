classdef FullyConnectedHostVectorStrategy < nnet.internal.cnn.layer.util.FullyConnectedVectorStrategy
    % FullyConnectedHostvectorStrategy   Execution strategy for running the
    % fully connected layer on the host with vector inputs

    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function X = sendToDevice(~, X)
            % No operation required for the host
        end
    end
end