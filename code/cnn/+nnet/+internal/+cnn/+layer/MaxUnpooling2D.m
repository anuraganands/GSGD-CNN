classdef MaxUnpooling2D < nnet.internal.cnn.layer.Layer
    % MaxUnpooling2D   Max 2D unpooling layer implementation
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();

        % Name (char array)   A name for the layer
        Name
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'maxunpool'
    end
            
    properties (SetAccess = private)
        % InputNames   This layer has three inputs
        InputNames = {'in','indices','size'}
        
        % OutputNames   This layer has one output
        OutputNames = {'out'}
        
        % HasSizeDetermined   Specifies if all size parameters are set
        %   This layer always has it's size determined.
        HasSizeDetermined = true               
    end
    
    methods
        function this = MaxUnpooling2D(name)
            this.Name = name;           
        end
        
        function Z = predict(~, inputsX)            
            % unpack inputs
            X          = inputsX{1};
            indices    = inputsX{2};
            outputSize = inputsX{3};
            
            % allocate output
            Z = zeros(outputSize, 'like', X);
            
            % unpool
            Z(indices) = X(1:numel(X));
        end
        
        function [dXOut,dW] = backward(~, inputsX, Z, dZ, ~)           
            % unpack inputs
            X       = inputsX{1};
            indices = inputsX{2};
            
            dX = zeros(size(X), 'like', Z);           
           
            dX(1:numel(X)) = dZ(indices);
            
            dXOut = {dX, [], []};
            
            dW = [];
        end

        function outputSize = forwardPropagateSize(~, sizesInCell)
            % Compute output size based on input parameters. 
            numChannels = sizesInCell{1}(3);
            
            % NB: The output size of this layer is expected to be the last
            % two elements of the third input size. This is special
            % information encoded by the max pooling layer.
            outputSize = [sizesInCell{3}(4:5) numChannels];
        end

        function this = inferSize(this, ~)
        end
        
        function tf = isValidInputSize(~, inputSizeInCellArray)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            
            tf = iscell(inputSizeInCellArray) && numel(inputSizeInCellArray) == 3;
            
            if ~tf
                return;
            end
            
            sizeX = inputSizeInCellArray{1};  
            sizeIndices = inputSizeInCellArray{2};
            sizeOutputSize = inputSizeInCellArray{3};
            
            % The first input is [H W C] feature map. Expect non-zero size. 
            tf = all(sizeX(1:3) > 0);
            
            % The second input is a column vector of indices.
            % Expected size is [L 1 1]. L should equal H*W*C.
            tf = tf ...
                && (sizeIndices(1) > 0) ...
                && all(sizeIndices(2:end) == 1) ...
                && sizeIndices(1) == prod(sizeX(1:3));
            
            % The third input is an output size. The fourth and fifth
            % elements are used to encode the input size of the max pooling
            % layer.
            hasFiveElements = numel(sizeOutputSize) == 5;
            tf = tf && hasFiveElements && isequal(sizeOutputSize(1:3), [1 4 1]); 
            
            % Unpooled output size should not be smaller than input.            
            tf = tf && all( sizeOutputSize(4:5) >= sizeX(1:2) );    
            
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
        end
        
        function this = setupForGPUPrediction(this)           
        end
        
        function this = setupForHostTraining(this)
        end
        
        function this = setupForGPUTraining(this)
        end
    end
end
