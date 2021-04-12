classdef LocalMapNorm2DGPUStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % LocalMapNorm2DGPUStrategy   Execution strategy for running local response normalization on the GPU  
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, windowSize, alpha, beta, k)
            Z = nnet.internal.cnngpu.localMapNormForward2D(X, windowSize, alpha, beta, k);
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ, X, windowSize, alpha, beta, k)
            dX =  nnet.internal.cnngpu.localMapNormBackward2D( ...
                Z, dZ, X, windowSize, alpha, beta, k);
            dW = [];
        end
    end
end