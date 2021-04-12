classdef BatchNormalizationHostStrategy
    % BatchNormalizationHostStrategy   Execution strategy for running batch normalization on the host
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forwardTrain(~, X, beta, gamma, epsilon)
            [Z,batchMean,batchInvVar] = ...
                nnet.internal.cnnhost.batchNormalizationForwardTrain(X, beta, gamma, epsilon);
            memory = {batchMean, batchInvVar};
        end
        
        function Z = forwardPredict(~, X, beta, gamma, epsilon, inputMean, inputVar)
            Z = nnet.internal.cnnhost.batchNormalizationForwardPredict(X, beta, gamma, epsilon, inputMean, inputVar);
        end
        
        function [dX,dW] = backward(~, ~, dZ, X, gamma, epsilon, memory)
            [batchMean, batchInvVar] = deal(memory{:});
            args = { dZ, X, gamma, epsilon, batchMean, batchInvVar };
            needsWeightGradients = nargout > 1;
            if ~needsWeightGradients
                dX = nnet.internal.cnnhost.batchNormalizationBackward( args{:} );
            else
                [dX,dW{1},dW{2}] = nnet.internal.cnnhost.batchNormalizationBackward( args{:} );
            end
        end
        
    end
end
