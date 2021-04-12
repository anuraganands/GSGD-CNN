classdef MaxPooling2D < nnet.internal.cnn.layer.Layer
    % MaxPooling2D   Max 2D pooling layer implementation
    
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
        DefaultName = 'maxpool'
    end
    
    properties (SetAccess = private)
        % InputNames   The layer has one input
        InputNames = {'in'}
        
        % HasSizeDetermined   Specifies if all size parameters are set
        %   Required to be false so that second output can be configured.
        HasSizeDetermined = false;
        
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
        
        % HasUnpoolingOutputs   Specifies whether this layer should have extra
        %   outputs that can be used for unpooling. This can be:
        %       true      - The layer will have two extra outputs 'indices'
        %                   and 'size' which can be used with a max
        %                   unpooling layer.
        %       false     - The layer will have one output 'out'.
        HasUnpoolingOutputs
    end
    
    properties(SetAccess = private, Dependent)
        % OutputNames   The number of outputs depends on whether
        %               'HasUnpoolingOutputs' is true or false
        OutputNames
    end
    
    properties(Access = private)
        ExecutionStrategy
    end
    
    methods
        function this = MaxPooling2D( ...
                name, poolSize, stride,  paddingMode, paddingSize, unpoolingOutputs)
            this.Name = name;
            
            % Set hyperparameters
            this.PoolSize = poolSize;
            this.Stride = stride;
            this.PaddingMode = paddingMode;
            this.PaddingSize = paddingSize;
            if nargin == 6
                this.HasUnpoolingOutputs = unpoolingOutputs;
            else
                this.HasUnpoolingOutputs = false;
            end
            
            this = selectHostStrategyBasedOnNumOutputs(this);
        end
        
        function Z = predict(this, X)
            % Note that padding is stored as [top bottom left right] but
            % the function expects [top left bottom right].
            Z = this.ExecutionStrategy.forward( ...
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
            outSize = [outputHeightAndWidth outputMaps];
            if this.HasUnpoolingOutputs
                % pooled feature map.
                outputSize{1} = outSize;
                
                % indices output.
                outputSize{2} = [prod([outputHeightAndWidth outputMaps]) 1 1];
                
                % output size.
                outputSize{3} = [1 4 1 inputSize(1) inputSize(2)];
                
                % NB: append input size so max unpooling can know what its
                % output size should be. Continue to put [1 4 1] as the
                % leading size because that is the size of this output.
                % Consumers of this size info can still rely on the first
                % three elements being the true size. Consumers that know
                % the last two elements contain the input size of the max
                % pooling layer can use that (i.e. max unpooling).
            else
                outputSize = outSize;
            end
        end
        
        function this = inferSize(this, inputSize)
            
            if iIsTheStringSame(this.PaddingMode)
                this.PaddingSize = iCalculateSamePadding( ...
                    this.PoolSize, this.Stride, inputSize(1:2));
            end
            
            if this.HasUnpoolingOutputs
                % If unpooling outputs are enabled, check that there are no
                % overlapping pooling regions
                poolingRegionsOverlap = any(this.Stride < this.PoolSize);
                
                if poolingRegionsOverlap
                    % Overlapping pooling windows are not supported when
                    % output indices are requested.
                    error(message('nnet_cnn:layer:MaxPooling2DLayer:IndicesRequireNonOverlappingPoolingRegion'));
                end
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
            this = selectHostStrategyBasedOnNumOutputs(this);
        end
        
        function this = setupForGPUPrediction(this)
            this = selectGPUStrategyBasedOnNumOutputs(this);
        end
        
        function this = setupForHostTraining(this)
            this = selectHostStrategyBasedOnNumOutputs(this);
        end
        
        function this = setupForGPUTraining(this)
            this = selectGPUStrategyBasedOnNumOutputs(this);
        end
        
        function val = get.OutputNames(this)
            if this.HasUnpoolingOutputs
                val = {'out','indices','size'};
            else
                val = {'out'};
            end
        end
    end
    
    methods(Access = private)
        function this = selectGPUStrategyBasedOnNumOutputs(this)
            % Switch execution strategy based on the outputs
            if this.HasUnpoolingOutputs
                this.ExecutionStrategy = nnet.internal.cnn.layer.util.MaxPooling2DWithOutputIndicesGPUStrategy();
            else
                this.ExecutionStrategy = nnet.internal.cnn.layer.util.MaxPooling2DGPUStrategy();
            end
        end
        
        function this = selectHostStrategyBasedOnNumOutputs(this)
            % Switch execution strategy based on the outputs
            if this.HasUnpoolingOutputs
                this.ExecutionStrategy = nnet.internal.cnn.layer.util.MaxPooling2DWithOutputIndicesHostStrategy();
            else
                this.ExecutionStrategy = nnet.internal.cnn.layer.util.MaxPooling2DHostStrategy();
            end
        end
    end
end

function heightAndWidthPadding = iCalculateHeightAndWidthPadding(paddingSize)
heightAndWidthPadding = nnet.internal.cnn.layer.padding.calculateHeightAndWidthPadding(paddingSize);
end

function tf = iIsTheStringSame(x)
tf = nnet.internal.cnn.layer.padding.isTheStringSame(x);
end

function paddingSize = iCalculateSamePadding(poolSize, stride, inputSize)
paddingSize = nnet.internal.cnn.layer.padding.calculateSamePadding(poolSize, stride, inputSize);
end