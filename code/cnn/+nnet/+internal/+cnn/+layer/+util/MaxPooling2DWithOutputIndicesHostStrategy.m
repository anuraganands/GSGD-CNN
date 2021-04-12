classdef MaxPooling2DWithOutputIndicesHostStrategy <nnet.internal.cnn.layer.util.ExecutionStrategy
    %  MaxPooling2DWithOutputIndicesHostStrategy  Execution strategy for
    %  running the max pooling with argmax indices on the Host.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(this, X, ...
                poolHeight, poolWidth, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride)
            
            M = nnet.internal.cnnhost.poolingMaxForward2D(X, ...
                poolHeight, poolWidth, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride);
            
            dZ = cast(reshape(1:numel(M), size(M)), 'like', M);
            
            argmax = nnet.internal.cnnhost.poolingMaxBackward2D(...
                M, dZ, X, ...
                poolHeight, poolWidth, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride);
            
            [indices, sz] = argmaxIndices(this, X, argmax);
            
            if numel(M) ~= numel(indices)
                % The number of indices must equal the number of elements
                % in the output feature map. They are not equal when the
                % input feature map contains NaN in the entire pooling
                % region.
                error(message('nnet_cnn:layer:MaxPooling2DLayer:NotEnoughIndices'));
            end
            
            Z = {M, indices, sz};
            
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ, X, ...
                poolHeight, poolWidth, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride)
            
            dX = nnet.internal.cnnhost.poolingMaxBackward2D(...
                Z{1}, dZ{1}, X, ...
                poolHeight, poolWidth, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride);
            dW = [];
        end   
        
        function [indices, sz] = argmaxIndices(~, X, argmax)
            % get indices.
            indices = find(argmax);
            indicesInOutput = argmax(indices);
            
            % sort indices so that the order matches the linear index order
            % of the output M.
            [~, ord] = sort(indicesInOutput);
            indices = indices(ord);
            
            % convert linear index to subscripts.
            [H, W, C, N] = size(X);
            sz = [H, W, C, N];
        end
        
    end
end