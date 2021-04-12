classdef(Abstract) FullyConnectedVectorStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % FullyConnectedVectorStrategy   Abstract base class for running the
    % fully connected layer with vector inputs

    %   Copyright 2017 The MathWorks, Inc.
    
    methods(Abstract)
        % Send data to desired hardware
        X = sendToDevice(this, X)
    end
    
    methods
        function [Z, memory] = forward(this, X, weights, bias)
            % Fold the input data
            [Xf, unfoldedDimensions] = this.foldDimensions(X);
            
            % Linear matrix multiplication
            Zf = this.sendToDevice(weights)*Xf + bias;
            
            % Unfold the output
            Z = this.unfoldDimensions(Zf, unfoldedDimensions);
            memory = [];
        end
        
        function [dX, dW] = backward(this, X, weights, dZ)
            % Fold the upper layer derivative
            [dZf, unfoldedDimensions] = this.foldDimensions(dZ);
            dZf = this.sendToDevice(dZf);
            
            % Data derivative
            dXf = weights'*dZf;
            dX = this.unfoldDimensions(dXf, unfoldedDimensions);
            
            needsWeightGradients = nargout > 1;
            if needsWeightGradients
                % Weights derivative
                Xf = this.foldDimensions(X);
                dW{1} = dZf*Xf';
                
                % Bias derivative
                dW{2} = sum( dZf, 2 );
            end
        end
    end
    
    methods(Access = private)
        function [Xf, unfoldedDimensions] = foldDimensions(~, X)
            % Fold dimensions greater than two into the second dimension
            sX = size(X);
            unfoldedDimensions = sX(2:end);
            Xf = reshape(X, [sX(1) prod(unfoldedDimensions)]);
        end
        
        function X = unfoldDimensions(~, Xf, unfoldedDimensions)
            % Return a folded array into its unfolded state
            s1 = size(Xf, 1);
            X = reshape(Xf, [s1 unfoldedDimensions]);
        end
    end
end