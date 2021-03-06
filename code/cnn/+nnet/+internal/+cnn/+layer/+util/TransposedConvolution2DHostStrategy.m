classdef TransposedConvolution2DHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % TransposedConvolution2DHostStrategy
    %     Execution strategy for running transposed convolution on the host
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, ...
                weights, bias, ...
                verticalPad, horizontalPad, ...
                verticalStride, horizontalStride)
            
            sz = nnet.internal.cnn.layer.TransposedConvolution2D.outputSize(X, weights, ...
                verticalPad, horizontalPad, verticalStride, horizontalStride);
            
            
            imageHeight = sz(1);
            imageWidth  = sz(2);
            if isa(X, 'single') && feature('UseMkldnn')
                Z = nnet.internal.cnnhost.convolveBackward2D(...
                    weights, X, ...
                    verticalPad, horizontalPad, ...
                    verticalPad, horizontalPad, ...
                    verticalStride, horizontalStride);
            else
                Z = nnet.internal.cnnhost.convolveBackwardData2DCore(...
                    [imageHeight, imageWidth], weights, X, ...
                    verticalPad, horizontalPad, ...
                    verticalPad, horizontalPad, ...
                    verticalStride, horizontalStride);
            end
            
            % add bias
            Z = Z + bias;
            
            memory = [];
            
        end
        
        function [dX, dW] = backward( ~, ...
                X, weights, dZ, ...
                verticalPad, horizontalPad, ...
                verticalStride, horizontalStride)
            needsWeightGradients = nargout > 1;
            
            if isa(X, 'single') && feature('UseMkldnn')
                dX = nnet.internal.cnnhost.convolveForward2D( ...
                    dZ, weights, ...
                    verticalPad, horizontalPad, ...
                    verticalPad, horizontalPad, ...
                    verticalStride, horizontalStride);
                
                if needsWeightGradients
                    [~, dW{1}] = nnet.internal.cnnhost.convolveBackward2D( ...
                        dZ, weights, X, ...
                        verticalPad, horizontalPad, ...
                        verticalPad, horizontalPad, ...
                        verticalStride, horizontalStride);
                    dW{2} = nnet.internal.cnnhost.convolveBackwardBias2D(dZ);
                end
            else
                dX = nnet.internal.cnnhost.stridedConv( ...
                    dZ, weights, ...
                    verticalPad, horizontalPad, ...
                    verticalPad, horizontalPad, ...
                    verticalStride, horizontalStride);
                
                if needsWeightGradients
                    dW{1} = nnet.internal.cnnhost.convolveBackwardFilter2D( ...
                        dZ, weights, X, ...
                        verticalPad, horizontalPad, ...
                        verticalPad, horizontalPad, ...
                        verticalStride, horizontalStride);
                    dW{2} = nnet.internal.cnnhost.convolveBackwardBias2D(dZ);
                end
            end
            
            
        end
        
    end
    
end
