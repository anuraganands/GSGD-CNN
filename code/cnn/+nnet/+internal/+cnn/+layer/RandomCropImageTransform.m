classdef RandomCropImageTransform < nnet.internal.cnn.layer.ImageTransform
    % RandomCropImageTransform   Image transform for random crops
    %   This class randomly crops an image to the size specified during
    %   construction. This size should equal the input size of the image input
    %   layer.
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties (Constant)
        Type = 'randcrop';
    end
    
    properties   
        % ImageSize   Size of the image that this transform will transform
        ImageSize
        
        % CropSize   Height and width of the cropped image.
        CropSize
        
        % NumChannels   Number of channels in an image.
        NumChannels
        
        % BatchIndex   Cell array of indices used to access a batch of
        % images.
        BatchIndex
    end
    
    methods
        
        %------------------------------------------------------------------
        % Return a random crop transform. Input is the size of the cropped
        % image and should equal the size of the input image layer when
        % used in a network.
        %------------------------------------------------------------------
        function this = RandomCropImageTransform(inputImageSize)
            this.ImageSize = inputImageSize;
            this.CropSize = inputImageSize(1:2);           
            
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

        function outputSize = forwardPropagateSize(this, inputSize)
            if(~this.inputSizeIsGreaterThanOrEqualToCropSize(inputSize))
                outputSize = NaN(1, 3);
            else
                outputSize = [this.CropSize inputSize(3)];
            end
        end
    end
    
    methods(Access = protected)
        
        %------------------------------------------------------------------
        % Transform implementation. Input is a batch of images. Output is a
        % batch of randomly cropped images.
        %------------------------------------------------------------------
        function y = doTransform(this, batch)
            
            % Size of images in batch.
            [M, N, P] = size(batch);
            
            numImagesInBatch = P/this.NumChannels;
            
            maxShift = [M N] - this.CropSize;
            
            % random translations for cropping
            rowShift = this.randomShift(maxShift(1), numImagesInBatch);
            colShift = this.randomShift(maxShift(2), numImagesInBatch);
            
            rowRange = bsxfun(@plus, 1:this.CropSize(1), rowShift);
            colRange = bsxfun(@plus, 1:this.CropSize(2), colShift);
            
            y = zeros(...
                [this.CropSize this.NumChannels numImagesInBatch], ...
                'like', batch);
            
            for i = 1:numImagesInBatch
                
                this.BatchIndex{end} = i;
                r = rowRange(i,:);
                c = colRange(i,:);
                
                % Crop image
                y(:,:,this.BatchIndex{:}) = batch(r,c, this.BatchIndex{:});
            end
        end
        
        %------------------------------------------------------------------
        % Return random shifts to apply to the image. Overloading this
        % method is useful for testing.
        %------------------------------------------------------------------
        function indices = randomShift(~, imax, N)
            indices = randi([0 imax], N, 1);
        end
        
        function tf = inputSizeIsGreaterThanOrEqualToCropSize(this, inputSize)
            tf = (inputSize(1) >= this.CropSize(1)) && (inputSize(2) >= this.CropSize(2));
        end
    end
end

