classdef(Abstract) FullyConnectedImageStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % FullyConnectedImageStrategy   Execution strategy for running the fully connected layer on the host

    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods(Abstract)
        % Send data to desired hardware
        X = sendToDevice(~, X)
        
        % Forward convolution
        Z = convolveForward(~, X, weights);
        
        % Backward data convolution
        dX = convolveBackwardData(X, weights, dZ)
        
        % Backward weights convolution
        dW = convolveBackwardFilter(X, weights, dZ)
        
        % Backward bias convolution
        dB = convolveBackwardBias(dZ)
    end
    
    methods
        function [Z, memory] = forward(this, X, weights, bias)
            Z = this.forwardConvolveOrMultiply(X, weights) + bias;
            memory = [];
        end
        
        function [dX, dW] = backward(this, X, weights, dZ)
            dX = this.backwardConvolveOrMultiply(X, weights, dZ);
            needsWeightGradients = nargout > 1;
            if needsWeightGradients
                dW{1} = this.backwardWeights(X, weights, dZ);
                dW{2} = this.backwardBias(X, weights, dZ);
            end
        end
    end
    
    methods(Access = private)
        function Z = forwardConvolveOrMultiply(this, X, weights)
            % If the first three dimensions of weights and X are the same,
            % use a simple multiply, otherwise use convolution
            [tf, szX, szW] = this.firstThreeDimsEqual(X, weights);
            if tf
                Z = reshape(this.sendToDevice(weights), [], szW(4))' * reshape(X, [], szX(4));
                Z = reshape(Z, 1, 1, szW(4), szX(4));
            else
                Z = this.convolveForward(X, weights);
            end
        end
        
        function dX = backwardConvolveOrMultiply(this, X, weights, dZ)
            % If the first three dimensions of weights and X are the same,
            % use a simple multiply, otherwise use convolution
            [tf, szX, szW] = this.firstThreeDimsEqual(X, weights);
            if tf
                dX = reshape(this.sendToDevice(weights), [], szW(4)) * reshape(dZ, szW(4), []);
                dX = reshape(dX, szX);
            else
                dX = this.convolveBackwardData(X, weights, dZ);
            end
        end
        
        function dW = backwardWeights(this, X, weights, dZ)
            % If the first three dimensions of weights and X are the same,
            % use a simple multiply, otherwise use convolution
            [tf, szX, szW] = this.firstThreeDimsEqual(X, weights);
            if tf
                dW = reshape(this.sendToDevice(X), [], szX(4)) * reshape(dZ, szW(4), szX(4))';
                dW = reshape(dW, szW);
            else
                dW = this.convolveBackwardFilter(X, weights, dZ);
            end
        end
        
        function dBias = backwardBias(this, X, weights, dZ)
            % If the first three dimensions of weights and X are the same,
            % use a simple sum, otherwise use convolution
            tf = this.firstThreeDimsEqual(X, weights);
            if tf
                dBias = sum(this.sendToDevice(dZ), 4);
            else
                dBias = this.convolveBackwardBias(dZ);
            end
        end
        
        function [tf, szX, szW] = firstThreeDimsEqual(this, X, W)
            szX = this.FourDSize(X);
            szW = this.FourDSize(W);
            tf = szX(1:3) == szW(1:3);
        end
        
        function sz = FourDSize(~, X)
            sz = ones(1,4);
            [sz(1), sz(2), sz(3), sz(4)] = size(X);
        end
    end
end