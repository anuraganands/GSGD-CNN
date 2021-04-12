classdef LeakyReLUGPUStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % LeakyReLUGPUStrategy   Execution strategy for running Leaky ReLU on 
    % the GPU
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, scale)
            Z = nnet.internal.cnngpu.leakyReluForward(X, scale);
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ, X, scale)
            dX = nnet.internal.cnngpu.leakyReluBackward(Z, dZ, X, scale);
            dW = [];
        end
    end
end