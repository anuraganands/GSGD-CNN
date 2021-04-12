classdef ReLUHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % ReLUHostStrategy   Execution strategy for running ReLU on the host
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X)
            Z = nnet.internal.cnnhost.reluForward(X);
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ, X)
            dX = nnet.internal.cnnhost.reluBackward(Z, dZ, X);
            dW = [];
        end
    end
end