classdef ClippedReLUGPUStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % ClippedReLUGPUStrategy   Execution strategy for running clipped ReLU on the GPU
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, ceiling)
            Z = nnet.internal.cnngpu.clippedReluForward(X, ceiling);
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ, X, ceiling)
            dX = nnet.internal.cnngpu.clippedReluBackward(Z, dZ, X, ceiling);
            dW = [];
        end
        
    end
end