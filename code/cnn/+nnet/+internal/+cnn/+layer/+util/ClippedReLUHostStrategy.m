classdef ClippedReLUHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % ClippedReLUHostStrategy   Execution strategy for running clipped ReLU on the host
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, ceiling)
            Z = nnet.internal.cnnhost.clippedReluForward(X, ceiling);
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ, X, ceiling)
            dX = nnet.internal.cnnhost.clippedReluBackward(Z, dZ, X, ceiling);
            dW = [];
        end
        
    end
end
