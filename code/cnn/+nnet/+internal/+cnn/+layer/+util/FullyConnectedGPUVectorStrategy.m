classdef FullyConnectedGPUVectorStrategy < nnet.internal.cnn.layer.util.FullyConnectedVectorStrategy
    % FullyConnectedHostvectorStrategy   Execution strategy for running the
    % fully connected layer on the GPU with vector inputs

    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function X = sendToDevice(~, X)
            X = gpuArray(X);
        end
    end
end