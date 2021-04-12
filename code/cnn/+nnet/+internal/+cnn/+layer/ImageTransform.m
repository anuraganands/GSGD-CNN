classdef ImageTransform < matlab.mixin.Heterogeneous
    % ImageTransform   Abstract image transform class.

    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties(Abstract, Constant)
        % Type   (array of char) The type corresponding to a transformation
        Type
    end
    
    properties(Abstract)
        % ImageSize   Size of the image that this transform will transform
        ImageSize
    end
    
    methods(Abstract, Access = protected)
        %------------------------------------------------------------------
        % doTransform Implements an image transformation. Implementations
        % must input a batch of RGB or grayscale images and output a batch
        % of images. 
        %------------------------------------------------------------------
        doTransform(this)
    end
    
    methods(Abstract)
        %------------------------------------------------------------------
        % serialize   Serialize an image transform to a structure 
        %------------------------------------------------------------------
        S = serialize( this )
    end
    
    methods
        %------------------------------------------------------------------
        % Returns true if the input size is for an RGB image. The input
        % image layer may have sizes specified as 
        % 
        %   [M N], [M N 1] for grayscale
        %   [M N 3] for RGB
        %------------------------------------------------------------------
        function tf = isRGBImage(~, inputImageSize)
            tf = numel(inputImageSize) == 3 ...
                && inputImageSize(end) == 3;
        end
    end

    methods(Abstract)
        % forwardPropagateSize   Calculate the output size for this transform
        %   Calculate the output size for this transform based on the input
        %   size. This is needed to check if training images have the
        %   correct dimensions. inputSize is a 3-element vector which is
        %   the size of the input to this layer.
        outputSize = forwardPropagateSize(this, inputSize)
    end
    
    methods(Sealed)
        %------------------------------------------------------------------
        % Applies an array of transforms to a batch of RGB or grayscale
        % images.
        %------------------------------------------------------------------
        function y = apply(this, batch)           
            y = batch;
            for i = 1:numel(this)
                y = doTransform(this(i), y);
            end
        end
    end
end