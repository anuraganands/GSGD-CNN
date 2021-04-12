classdef AveragePooling2D < nnet.internal.cnn.layer.Layer
    % AveragePooling2D   Average 2D pooling layer implementation
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();

        % Name (char array)   A name for the layer
        Name
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'avgpool'
    end
        
    properties (SetAccess = private)
        % InputNames   This layer has a single input
        InputNames = {'in'}
        
        % OutputNames   This layer has a single output
        OutputNames = {'out'}
        
        % HasSizeDetermined   Specifies if all size parameters are set
        HasSizeDetermined

        % PoolSize   The height and width of a pooling region
        %   The size the pooling regions. This is a vector [h w] where h is
        %   the height of a pooling region, and w is the width of a pooling
        %   region.
        PoolSize
        
        % Stride   The vertical and horizontal stride
        %   The step size for traversing the input vertically and
        %   horizontally. This is a vector [u v] where u is the vertical
        %   stride and v is the horizontal stride.
        Stride
        
        % PaddingMode   The mode used to determine the padding
        %   The mode used to calculate the PaddingSize property. This can
        %   be:
        %       'manual'    - PaddingSize is specified manually. 
        %       'same'      - PaddingSize will be calculated so that the 
        %                     output size is the same size as the input 
        %                     when the stride is 1. More generally, the 
        %                     output size will be ceil(inputSize/stride),
        %                     where inputSize is the height and width of 
        %                     the input.
        PaddingMode
        
        % PaddingSize   The padding applied to the input along the edges
        %   The padding that is applied along the edges. This is a row 
        %   vector [t b l r] where t is the padding to the top, b is the 
        %   padding applied to the bottom, l is the padding applied to the 
        %   left, and r is the padding applied to the right.
        PaddingSize
    end
    
    properties(Access = private)
        ExecutionStrategy
    end
    
    methods
        function this = AveragePooling2D( ...
                name, poolSize, stride, paddingMode, paddingSize)
            this.Name = name;
            
            % Size is determined if padding mode is not 'same'.
            this.HasSizeDetermined = ~iIsTheStringSame(paddingMode);
            
            % Set hyperparameters
            this.PoolSize = poolSize;
            this.Stride = stride;
            this.PaddingMode = paddingMode;
            this.PaddingSize = paddingSize;
            
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.AveragePooling2DHostStrategy();
        end
        
        function Z = predict( this, X )
            % Note that padding is stored as [top bottom left right] but
            % the function expects [top left bottom right].
            Z = this.ExecutionStrategy.forward(...
                X, ...
                this.PoolSize(1), this.PoolSize(2), ...
                this.PaddingSize(1), this.PaddingSize(3), ...
                this.PaddingSize(2), this.PaddingSize(4), ...
                this.Stride(1), this.Stride(2));
        end
        
        function [dX,dW] = backward(this, X, Z, dZ, ~)
            % Note that padding is stored as [top bottom left right] but
            % the function expects [top left bottom right].
            [dX,dW] = this.ExecutionStrategy.backward(...
                Z, dZ, X, ...
                this.PoolSize(1), this.PoolSize(2), ...
                this.PaddingSize(1), this.PaddingSize(3), ...
                this.PaddingSize(2), this.PaddingSize(4), ...
                this.Stride(1), this.Stride(2));
        end

        function outputSize = forwardPropagateSize(this, inputSize)
            heightAndWidthPadding = iCalculateHeightAndWidthPadding(this.PaddingSize);
            outputHeightAndWidth = floor((inputSize(1:2) + heightAndWidthPadding - this.PoolSize)./this.Stride) + 1;
            outputMaps = inputSize(3);
            outputSize = [outputHeightAndWidth outputMaps];
        end

        function this = inferSize(this, inputSize)
            if iIsTheStringSame(this.PaddingMode)
                this.PaddingSize = iCalculateSamePadding( ...
                    this.PoolSize, this.Stride, inputSize(1:2));
                
                % If the padding is set to 'same', the size will always
                % need to be determined again because we will need to
                % recalculate the padding.
                this.HasSizeDetermined = false;
            else
                this.HasSizeDetermined = true;
            end
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            heightAndWidthPadding = iCalculateHeightAndWidthPadding(this.PaddingSize);
            tf = all( this.PoolSize <= inputSize(1:2) + heightAndWidthPadding );
        end        

        function this = initializeLearnableParameters(this, ~)
        end
        
        function this = prepareForTraining(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.empty();
        end
        
        function this = prepareForPrediction(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        end
        
        function this = setupForHostPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.AveragePooling2DHostStrategy();
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.AveragePooling2DGPUStrategy();
        end
        
        function this = setupForHostTraining(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.AveragePooling2DHostStrategy();
        end
        
        function this = setupForGPUTraining(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.AveragePooling2DGPUStrategy();
        end
    end
end

function tf = iIsTheStringSame(x)
tf = nnet.internal.cnn.layer.padding.isTheStringSame(x);
end

function heightAndWidthPadding = iCalculateHeightAndWidthPadding(paddingSize)
heightAndWidthPadding = nnet.internal.cnn.layer.padding.calculateHeightAndWidthPadding(paddingSize);
end

function paddingSize = iCalculateSamePadding(poolSize, stride, inputSize)
paddingSize = nnet.internal.cnn.layer.padding.calculateSamePadding(poolSize, stride, inputSize);
end