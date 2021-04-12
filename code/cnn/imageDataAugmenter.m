classdef imageDataAugmenter < handle
    % imageDataAugmenter Configure image data augmentation
    %
    %   aug = imageDataAugmenter() creates an imageDataAugmenter object
    %   with default property values. The default state of the
    %   imageDataAugmenter is the identity transformation.
    %
    %   aug = imageDataAugmenter(Name,Value,___) configures a set of image
    %   augmentation options using Name/Value pairs to set properties.
    %
    %   imageDataAugmenter properties:
    %       FillValue           - Value used to define out of bounds points
    %       RandXReflection     - Random X reflection
    %       RandYReflection     - Random Y reflection
    %       RandRotation        - Random rotation
    %       RandXScale          - Random X scale
    %       RandYScale          - Random Y scale
    %       RandXShear          - Random X shear
    %       RandYShear          - Random Y shear
    %       RandXTranslation    - Random X translation
    %       RandYTranslation    - Random Y translation
    %
    %   Example 1
    %   ---------
    %   Train a convolutional neural network on some synthetic images of
    %   handwritten digits. Apply random rotations during training to add
    %   rotation invariance to trained network.
    %
    %   [XTrain, YTrain] = digitTrain4DArrayData;
    %
    %   imageSize = [28 28 1];
    %
    %   layers = [ ...
    %       imageInputLayer(imageSize,'Normalization','none');
    %       convolution2dLayer(5,20);
    %       reluLayer();
    %       maxPooling2dLayer(2,'Stride',2);
    %       fullyConnectedLayer(10);
    %       softmaxLayer();
    %       classificationLayer()];
    %
    %   opts = trainingOptions('sgdm','Plots','training-progress');
    %
    %   imageAugmenter = imageDataAugmenter('RandRotation',[-10 10]);
    %
    %   datasource = augmentedImageSource(imageSize,XTrain,YTrain,'DataAugmentation',imageAugmenter);
    %
    %   net = trainNetwork(datasource,layers,opts);
    %
    % See also augmentedImageSource, imageInputLayer, trainNetwork

    % Copyright 2017 The MathWorks, Inc.
    
    properties (SetAccess = 'private')
        
        %FillValue - Value used to define out of bounds points when resampling
        %
        %    FillValue is a numeric scalar or vector. If images being
        %    augmented are single channel, FillValue must be scalar. If
        %    images being augmented are multichannel then FillValue may be
        %    a scalar or a vector with length equal to the number of
        %    channels in the input image data.
        FillValue
        
        %RandXReflection - Random X reflection
        %
        %    RandXReflection is a logical scalar that specifies whether
        %    random left/right reflections are applied to input image data.
        %
        %    Default: false
        %
        RandXReflection
        
        %RandYReflection - Random Y reflection
        %
        %    RandYReflection is a logical scalar that specifies whether
        %    random up/down reflections are applied to input image data.
        %
        %    Default: false
        %
        RandYReflection
        
        %RandRotation - Random rotation
        %
        %    RandRotation is a two element numeric vector that specifies
        %    the random range, in degrees, of rotations that are applied to
        %    input image data.
        %
        %    Default: [0 0]
        RandRotation
        
        %RandXScale - Random X scale
        %
        %    RandXScale is a two element numeric vector that specifies the
        %    random range of scale in the X dimension that is applied to
        %    input image data.
        %
        %    Default: [1 1]
        RandXScale
        
        %RandYScale - Random Y scale
        %
        %    RandYScale is a two element numeric vector that specifies the
        %    random range of scale in the Y dimension that is applied to
        %    input image data.
        %
        %    Default: [1 1]
        RandYScale
        
        %RandXShear - Random X shear
        %
        %    RandXShear is a two element numeric vector that specifies the
        %    random range of shear in the X dimension that is applied to
        %    input image data. Shear is specified in terms of shear angles
        %    measured in units of degres. The valid range of shear angles
        %    is (-90,90) degrees.
        %
        %    Default: [0 0]
        RandXShear
        
        %RandYShear - Random Y shear
        %
        %    RandYShear is a two element numeric vector that specifies the
        %    random range of shear in the Y dimension that is applied to
        %    input image data.  Shear is specified in terms of shear angles
        %    measured in units of degres. The valid range of shear angles
        %    is (-90,90) degrees.
        %
        %    Default: [0 0]
        RandYShear
        
        %RandXTranslation - Random X translation
        %
        %    RandXTranslation is a two element numeric vector that
        %    specifies the random range of translation in the X dimension,
        %    in units of pixels, that is applied to input image data.
        %
        %    Default: [0 0]
        RandXTranslation
        
        %RandYTranslation - Random Y translation
        %
        %    RandYTranslation is a two element numeric vector that
        %    specifies the random range of translation in the Y dimension,
        %    in units of pixels, that is applied to input image data.
        %
        %    Default: [0 0]
        RandYTranslation
                
    end
    
    % Public, cache intermediate state of transform to allow for testing
    properties (Hidden)
        
        Rotation
        XReflection
        YReflection
        XScale
        YScale
        XShear
        YShear
        XTranslation
        YTranslation
        
        AffineTransforms
        
    end
    
    methods
        
        function self = imageDataAugmenter(varargin)
            %imageDataAugmenter Construct imageDataAugmenter object.
            %
            %   augmenter = imageDataAugmenter() constructs an
            %   imageDataAugmenter object with default property settings.
            %
            %   augmenter = imageDataAugmenter('Name',Value,___) specifies
            %   parameters that control aspects of data augmentation.
            %   Parameter names can be abbreviated and case does not
            %   matter.
            %
            %   Parameters include:
            %
            %   'FillValue'         A numeric scalar or vector 
            %                       (when augmenting multi-channel images) that
            %                       defines the value used during resampling
            %                       when transformed points fall out of bounds.
            %
            %                       Default: 0
            %
            %   'RandXReflection'   A scalar logical that defines whether
            %                       random left/right reflections are
            %                       applied.
            %
            %                       Default: false
            %
            %   'RandYReflection'   A scalar logical that defines whether
            %                       random up/down reflections are applied.
            %
            %                       Default: false
            %
            %   'RandRotation'      A two element vector that defines the
            %                       uniform range, in units of degrees, of
            %                       rotations that will be applied.
            %
            %                       Default: [0 0]
            %
            %   'RandXScale'        A two element vector that defines the
            %                       uniform range of scale that will
            %                       applied in the X dimension.
            %
            %                       Default: [1 1]
            %
            %   'RandYScale'        A two element vector that defines the
            %                       uniform range of scale that will be
            %                       applied in the Y dimension.
            %
            %                       Default: [1 1]
            %
            %   'RandXShear'        A two element vector that defines the
            %                       uniform range of shear that will be
            %                       applied in the X dimension. Specified
            %                       as a shear angle in units of degrees.
            %
            %                       Default: [0 0]
            %
            %   'RandYShear'        A two element vector that defines the
            %                       uniform range of shear that will be
            %                       applied in the Y dimension. Specified
            %                       as a shear angle in units of degrees.
            %
            %                       Default: [0 0]
            %
            %   'RandXTranslation'  A two element vector that defines the
            %                       uniform range of translation that will
            %                       be applied in the X dimension. Specified
            %                       in units of pixels.
            %
            %                       Default: [0 0]
            %
            %   'RandYTranslation'  A two element vector that defines the
            %                       uniform range of translation that will
            %                       be applied in the Y dimension. Specified
            %                       in units of pixels.
            %
            %                       Default: [0 0]
            
            self.parseInputs(varargin{:});
            
        end
        
    end
    
    % set methods for input validation
    methods
        
        function set.RandXReflection(self,xReflect)
            
            validateattributes(xReflect,{'numeric','logical'},{'real','scalar','nonempty'},mfilename,'RandXReflection');
            self.RandXReflection = logical(xReflect);
            
        end
        
        function set.RandYReflection(self,yReflect)
            
            validateattributes(yReflect,{'numeric','logical'},{'real','scalar','nonempty'},mfilename,'RandYReflection');
            self.RandYReflection = logical(yReflect);
            
        end
        
        function set.RandRotation(self,rotationInDegrees)
            
            validateattributes(rotationInDegrees,{'numeric'},{'real','finite','vector','numel',2},mfilename,'RandRotation');
            validateRange(rotationInDegrees,'RandRotation');
            self.RandRotation = rotationInDegrees;
            
        end
        
        function set.RandXScale(self,xScale)
            
            validateattributes(xScale,{'numeric'},{'real','finite','positive','vector','numel',2},mfilename,'RandXScale');
            validateRange(xScale,'RandXScale');
            self.RandXScale = xScale;
            
        end
        
        function set.RandYScale(self,yScale)
            
            validateattributes(yScale,{'numeric'},{'real','finite','positive','vector','numel',2},mfilename,'RandYScale');
            validateRange(yScale,'RandYScale');
            self.RandYScale = yScale;
            
        end
        
        function set.RandXShear(self,xShear)
            
            validateattributes(xShear,{'numeric'},{'real','finite','vector','numel',2,'>=',-90,'<=',90},mfilename,'RandXShear');
            validateRange(xShear,'RandXShear');
            self.RandXShear = xShear;
            
        end
        
        function set.RandYShear(self,yShear)
            
            validateattributes(yShear,{'numeric'},{'real','finite','vector','numel',2,'>=',-90,'<=',90,},mfilename,'RandYShear');
            validateRange(yShear,'RandYShear');
            self.RandYShear = yShear;
            
        end
        
        function set.RandXTranslation(self,xTrans)
            
            validateattributes(xTrans,{'numeric'},{'real','finite','vector','numel',2},mfilename,'RandXTranslation');
            validateRange(xTrans,'RandXTranslation');
            self.RandXTranslation = xTrans;
            
        end
        
        function set.RandYTranslation(self,yTrans)
            
            validateattributes(yTrans,{'numeric'},{'real','finite','vector','numel',2},mfilename,'RandYTranslation');
            validateRange(yTrans,'RandYTranslation');
            self.RandYTranslation = yTrans;
            
        end
                
        function set.FillValue(self,fillIn)
            validateattributes(fillIn,{'numeric'},{'real','vector'},mfilename,'FillValue');
            
            if (length(fillIn) ~= 1) && (length(fillIn) ~= 3)
                error(message('nnet_cnn:imageDataAugmenter:invalidFillValue'));
            end
            
            self.FillValue = fillIn;
        end
        
    end
    
    methods (Access = 'private')
        
        function Tout = makeAffineTransform(self,inputSize)
            
            if self.RandXReflection
                xReflect = sign(rand - 0.5);
            else
                xReflect = 1;
            end
            
            if self.RandYReflection
                yReflect = sign(rand - 0.5);
            else
                yReflect = 1;
            end
            
            self.XReflection = xReflect;
            self.YReflection = yReflect;
            self.Rotation = selectUniformRandValueFromRange(self.RandRotation);
            self.XScale = selectUniformRandValueFromRange(self.RandXScale);
            self.YScale = selectUniformRandValueFromRange(self.RandYScale);
            self.XShear = selectUniformRandValueFromRange(self.RandXShear);
            self.YShear = selectUniformRandValueFromRange(self.RandYShear);
            self.XTranslation = selectUniformRandValueFromRange(self.RandXTranslation);
            self.YTranslation = selectUniformRandValueFromRange(self.RandYTranslation);
            
            centerXShift = (inputSize(2)-1)/2;
            centerYShift = (inputSize(1)-1)/2;
            
            [moveToOriginTransform,moveBackTransform] = deal(eye(3));
            moveToOriginTransform(3,1) = -centerXShift;
            moveToOriginTransform(3,2) = -centerYShift;
            moveBackTransform(3,1) = centerXShift;
            moveBackTransform(3,2) = centerYShift;
             
            centeredRotation = moveToOriginTransform * self.makeRotationTransform() * moveBackTransform;
            centeredShear = moveToOriginTransform * self.makeShearTransform() * moveBackTransform;
            centeredScale = moveToOriginTransform * self.makeScaleTransform() * moveBackTransform;
            centeredReflection = moveToOriginTransform * self.makeReflectionTransform() * moveBackTransform;
                                    
            Tout = centeredRotation * centeredShear * centeredScale * centeredReflection * self.makeTranslationTransform();
              
        end
        
        function parseInputs(self,varargin)
            
            p = inputParser();
            p.addParameter('FillValue',0);
            p.addParameter('RandXReflection',false);
            p.addParameter('RandYReflection',false);
            p.addParameter('RandRotation',[0 0]);
            p.addParameter('RandXScale',[1, 1]);
            p.addParameter('RandYScale',[1, 1]);
            p.addParameter('RandXShear',[0, 0]);
            p.addParameter('RandYShear',[0, 0]);
            p.addParameter('RandXTranslation',[0,0]);
            p.addParameter('RandYTranslation',[0,0]);
            
            p.parse(varargin{:});
            params = p.Results;
            
            self.FillValue = params.FillValue;
            self.RandXReflection = params.RandXReflection;
            self.RandYReflection = params.RandYReflection;
            self.RandRotation = params.RandRotation;
            self.RandXScale = params.RandXScale;
            self.RandYScale = params.RandYScale;
            self.RandXShear = params.RandXShear;
            self.RandYShear = params.RandYShear;
            self.RandXTranslation = params.RandXTranslation;
            self.RandYTranslation = params.RandYTranslation;
            
        end
        
        function tform = makeRotationTransform(self)
                        
            tform = [cosd(self.Rotation), -sind(self.Rotation), 0;...
                sind(self.Rotation), cosd(self.Rotation), 0;...
                0, 0, 1];
            
        end
        
        function tform = makeShearTransform(self)
            
            tform = [1, tand(self.YShear), 0;...
                tand(self.XShear), 1, 0;...
                0, 0, 1];
            
        end
        
        function tform = makeScaleTransform(self)
            
            tform = [self.XScale, 0, 0;...
                0, self.YScale, 0;...
                0, 0, 1];
            
        end
        
        function tform = makeReflectionTransform(self)
            
            tform = [self.XReflection, 0, 0;...
                0, self.YReflection, 0;...
                0, 0, 1];
            
        end
        
        function tform = makeTranslationTransform(self)
           tform = eye(3);
           tform(3,1) = self.XTranslation;
           tform(3,2) = self.YTranslation;
        end
        
    end
    
    methods (Hidden)
        
        function B = augment(self,A)
            %augment Augment input image data.
            %
            %   augmentedImage = augment(augmenter,A) performs image
            %   augmentation on the input image A. A is an MxNxC matrix.
            %
            %   augmentedImageBatch = augment(augmenter,miniBatch) performs
            %   image augmentation on the input image batch miniBatch.
            %   miniBatch is a B element cell array containing MxNxC
            %   images.
                
            if iscell(A)    
                B = cell(size(A));
                self.AffineTransforms = zeros(3,3,length(A)); % 3 x 3 x batchSize matrix
                for img = 1:numel(A)
                    fillValue = manageFillValue(A{img},self.FillValue);
                    tform = self.makeAffineTransform(size(A{img}));
                    B{img} = nnet.internal.cnnhost.warpImage2D(A{img},tform(1:3,1:2),'linear',fillValue);
                    self.AffineTransforms(:,:,img) = tform;
                end
                
            elseif isnumeric(A) && (ndims(A) < 4)
                tform = self.makeAffineTransform(size(A));
                B = nnet.internal.cnnhost.warpImage2D(A,tform(1:3,1:2),'linear',manageFillValue(A,self.FillValue));
                self.AffineTransforms = tform;
            else
                assert(false,'Invalid image data input to augment.');
            end
            
        end
        
        function [A, B] = augmentPair(self, X, Y, interpY, fillValueY)
            %augmentPair Augment inputs X and Y.
            %
            %   [A, B] = augmentPair(augmenter, X, Y) performs the same
            %   image augmentation on the input image X and Y. X and Y are
            %   MxNxC matrices or B element cell arrays containing MxNxC
            %   iamges.
            %
            %   [...] = augmentPair(augmenter, X, Y, interpY, fillValueY)
            %   optionally specify the interpolation method and fill value
            %   to use for augmenting Y. By default, interpY is 'linear'
            %   and fillValueY is augmenter.FillValue.
                                 
            if nargin < 5
                fillValueY = self.FillValue;
            end
            
            if nargin < 4
                interpY = 'linear';                
            end
                        
            A = self.augment(X);
            tform = self.AffineTransforms;            
            B = self.augmentY(Y, tform, interpY, fillValueY);                        
            
        end
        
        function B = augmentY(~, Y, tform, interp, fillValue)
            % Augment using known transform.
            if iscell(Y)
                
                B = cell(size(Y));
                for img = 1:numel(Y)   
                    fillValue = manageFillValue(Y{img},fillValue);
                    B{img} = nnet.internal.cnnhost.warpImage2D(Y{img},tform(1:3,1:2,img),interp,fillValue);
                end
                
            elseif isnumeric(Y) && (ndims(Y) < 4)               
                B = nnet.internal.cnnhost.warpImage2D(Y,tform(1:3,1:2),interp, manageFillValue(Y,fillValue));
            else
                assert(false,'Invalid image data input to augment.');
            end
        end
        
    end
    
    methods(Static, Hidden = true)
        function self = loadobj(S)
            self = imageDataAugmenter('FillValue',S.FillValue,...
                'RandXReflection',S.RandXReflection,...
                'RandYReflection',S.RandYReflection,...
                'RandRotation',S.RandRotation,...
                'RandXScale',S.RandXScale,...
                'RandYScale',S.RandYScale,...
                'RandXShear',S.RandXShear,...
                'RandYShear',S.RandYShear,...
                'RandXTranslation',S.RandXTranslation,...
                'RandYTranslation',S.RandYTranslation);
                
        end
    end
    
    methods (Hidden)
        function S = saveobj(self)
            % Serialize denoisingImageDatasource object
            S = struct('FillValue',self.FillValue,...
                'RandXReflection',self.RandXReflection,...
                'RandYReflection',self.RandYReflection,...
                'RandRotation',self.RandRotation,...
                'RandXScale',self.RandXScale,...
                'RandYScale',self.RandYScale,...
                'RandXShear',self.RandXShear,...
                'RandYShear',self.RandYShear,...
                'RandXTranslation',self.RandXTranslation,...
                'RandYTranslation',self.RandYTranslation);    
        end
        
    end
       
end

function fillVal = manageFillValue(A,fillValIn)

if (size(A,3) > 1) && isscalar(fillValIn)
    fillVal = repmat(fillValIn,[1 size(A,3)]);
else
    fillVal = fillValIn;
end

end

function val = selectUniformRandValueFromRange(range)

val = diff(range) * rand + range(1);

end

function validateRange(range,propName)

if range(1) > range(2)
    error(message('nnet_cnn:imageDataAugmenter:invalidRange',propName));
end

end
