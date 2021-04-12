classdef SoftmaxGPUImageStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % SoftmaxGPUImageStrategy   Execution strategy for running the softmax
    % layer on the GPU with image inputs

    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X)
            Z = nnet.internal.cnngpu.softmaxForward2D(X);
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ)
            Z = nnet.internal.cnn.util.boundAwayFromZero(Z);
            dX = nnet.internal.cnngpu.softmaxBackward2D(Z, dZ);
            dW = [];
        end
    end
end