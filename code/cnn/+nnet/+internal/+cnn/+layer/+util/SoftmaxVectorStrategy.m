classdef(Abstract) SoftmaxVectorStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % SoftmaxVectorStrategy   Abstract execution strategy for running the
    % softmax layer with vector inputs

    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X)
            exponents = X - max(X,[],1);
            expX = exp(exponents);
            Z = expX./sum(expX);
            memory = [];
        end
        
        function [dX, dW] = backward(~, Z, dZ)
            Z = nnet.internal.cnn.util.boundAwayFromZero(Z);
            dotProduct = sum(Z.*dZ);
            dX = dZ - dotProduct;
            dX = dX.*Z;
            dW = [];
        end
    end
end