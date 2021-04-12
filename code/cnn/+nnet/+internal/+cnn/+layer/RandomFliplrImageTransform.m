classdef RandomFliplrImageTransform < nnet.internal.cnn.layer.ImageTransform
    % RandonFliplrImageTransform   Image transformation for random vetical flips
    %   This class randomly flips an image along the vertical axis (i.e. 
    %   fliplr)
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties (Constant)
        Type = 'randfliplr';
    end
        
    properties
        % ImageSize   Size of the image that this transform will transform
        ImageSize
        
        % NumChannels   Number of channels in an image.
        NumChannels
        
        % BatchIndex Cell array of indices used to access a batch of
        % images.
        BatchIndex
    end
    
    methods
        %------------------------------------------------------------------
        % Returns a random flip transform. Input to constructor is the size
        % of the image, which should match the image size set in the image
        % input layer. 
        %------------------------------------------------------------------             
        function this = RandomFliplrImageTransform(inputImageSize) 
            this.ImageSize = inputImageSize;        
            
            if isRGBImage(this, inputImageSize) 
                this.NumChannels = 3;                                
            else
                this.NumChannels = 1;                          
            end
            this.BatchIndex = {1:this.NumChannels, []};
        end
        
        %------------------------------------------------------------------
        % serialize   Serialize an image transform to a structure 
        %------------------------------------------------------------------
        function S = serialize( this )
            S.Version = 1.0;
            S.Type = this.Type;
            S.ImageSize = this.ImageSize;
        end

        function outputSize = forwardPropagateSize(~, inputSize)
            outputSize = inputSize;
        end
    end
    
    methods(Access = protected)
        %------------------------------------------------------------------
        % Transform implementation. Input is a batch of images. Output is a
        % batch of images where each image is randomly flipped.
        %------------------------------------------------------------------
        function batch = doTransform(this, batch)
            
            % Size of images in batch.
            [~, ~, P] = size(batch);
            
            numImagesInBatch = P/this.NumChannels;
            
            flip = this.randomFlip(numImagesInBatch);
            this.BatchIndex{end} = flip;
            
            batch(:,:,this.BatchIndex{:}) = fliplr(batch(:,:,this.BatchIndex{:}));
        end
        
        function ind = randomFlip(~, N)
            ind = logical(randi([0 1], N, 1));
        end
    end
end

