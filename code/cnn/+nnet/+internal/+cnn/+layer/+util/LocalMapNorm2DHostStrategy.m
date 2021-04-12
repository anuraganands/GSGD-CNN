classdef LocalMapNorm2DHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % LocalMapNorm2DHostStrategy   Execution strategy for running local response normalization on the host
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, windowSize, alpha, beta, k)
            Z = nnet.internal.cnnhost.localMapNormForward2D(X, windowSize, alpha, beta, k);
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ, X, windowSize, alpha, beta, k)
            dX =  nnet.internal.cnnhost.localMapNormBackward2D( ...
                Z, dZ, X, windowSize, alpha, beta, k);
            dW = [];
        end
    end
end