classdef ZeroCenterImageTransform < nnet.internal.cnn.layer.ImageTransform
    % ZeroCenterImageTransform   Image transform for zero centering
    %   This class zero centers any input images by subtracting the mean
    %   image.
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties (Constant)
        Type = 'zerocenter';
    end
    
    properties
        % ImageSize   Size of the image that this transform will transform
        ImageSize
    end
    
    properties(Dependent)
        % AverageImage   Average image
        AverageImage
    end
    
    properties(Access = private)
        % PrivateAverageImage   Private storage for public AverageImage
        % property
        PrivateAverageImage = [];
        
        % MeanPerChannel   Mean per-channel of the average image. Size is
        % 1x1xC, where C is the number of channels for the average image.
        MeanPerChannel = [];
    end
    
    methods
        function this = ZeroCenterImageTransform(inputImageSize)
            this.ImageSize = inputImageSize;
        end
        
        function S = serialize( this )
            S.Version = 1.0;
            S.Type = this.Type;
            S.ImageSize = this.ImageSize;
            S.AverageImage = this.AverageImage;
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            outputSize = inputSize;
        end
        
        function this = set.AverageImage(this, averageImage)
            this.PrivateAverageImage = averageImage;
            this.MeanPerChannel = iMeanPerChannel( averageImage );
        end
        
        function averageImage = get.AverageImage(this)
            averageImage = this.PrivateAverageImage;
        end
    end
    
    methods(Access = protected)
        function batch = doTransform(this, batch)
            if isequal( iHorizontalAndVerticalSize( batch ), this.ImageSize(1:2) )
                assert(~isempty(this.AverageImage));
                batch = batch - this.AverageImage;
            else
                assert(~isempty(this.MeanPerChannel));
                batch = batch - this.MeanPerChannel;
            end
        end
    end
end

function res = iHorizontalAndVerticalSize( X )
res = [size(X, 1) size(X, 2)];
end

function meanPerChannel = iMeanPerChannel( img )
% iMeanPerChannel   Compute the mean per-channel of image img.
% meanPerChannel is a 1x1xC vector, where C is the number of channels for
% img.
if isempty( img )
    meanPerChannel = [];
else
    numChannels = size(img, 3);
    meanPerChannel = mean( reshape( img, [], 1, numChannels ) );
end
end