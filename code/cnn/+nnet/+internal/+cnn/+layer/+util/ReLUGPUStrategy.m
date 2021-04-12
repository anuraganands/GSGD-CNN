classdef ReLUGPUStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % ReLUGPUStrategy   Execution strategy for running ReLU on the GPU
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X)
            Z = nnet.internal.cnngpu.reluForward(X);
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ, X)
            dX = nnet.internal.cnngpu.reluBackward(Z, dZ, X);
            dW = [];
        end
    end
end