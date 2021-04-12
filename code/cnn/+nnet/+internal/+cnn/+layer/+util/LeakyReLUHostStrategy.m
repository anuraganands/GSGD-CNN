classdef LeakyReLUHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % LeakyReLUHostStrategy   Execution strategy for running Leaky ReLU on 
    % the host
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, scale)
            Z = nnet.internal.cnnhost.leakyReluForward(X, scale);
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ, X, scale)
            dX = nnet.internal.cnnhost.leakyReluBackward(Z, dZ, X, scale);
            dW = [];
        end
    end
end