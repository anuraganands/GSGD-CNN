%   augmentedImageDatastore Generate batches of augmented image data
%
%   ds = augmentedImageDatastore(outputSize,imds) returns an
%   augmentedImageDatastore. outputSize is a two element vector which
%   specifies the output image size in the form [outputWidth,
%   outputHeight]. imds is an imageDatastore.
%
%   ds = augmentedImageDatastore(outputSize,X,Y) returns an
%   augmentedImageDatastore given matrices X and Y that define examples and 
%   corresponding responses.
%
%   ds = augmentedImageDatastore(X) returns an augmentedImageDatastore
%   given an input matrix X that defines examples.
%
%   ds = augmentedImageDatastore(outputSize,tbl) returns a
%   an augmentedImageDatastore given a tbl which contains predictors in the 
%   first column as either absolute or relative image paths or images. 
%   If responses are specified, responses must be in the second column as 
%   categorical labels for the images. In a regression problem, responses 
%   must be in the second column as either vectors or cell arrays containing 
%   3-D arrays or in multiple columns as scalars.
%
%   ds = augmentedImageDatastore(outputSize,tbl, responseName,___)
%   returns an augmentedImageDatastore which yields predictors and responses. 
%   tbl is a MATLAB table. responseName is a character vector specifying
%   the name of the variable in tbl that contains the responses.
%
%   ds = augmentedImageDatastore(outputSize,tbl,responseNames,___) returns
%   an augmentedImageDatastore for use in multi-output regression problems. 
%   tbl is a MATLAB table. responseNames is a cell array of character 
%   vectors specifying the names of the variables in tbl that contain the
%   responses.
%
%   ds = augmentedImageDatastore(___,Name,Value) returns an
%   augmentedImageDatastore using Name/Value pairs to configure
%   image-preprocessing options.
%
%   Parameters include:
%
%   'ColorPreprocessing'    A scalar string or character vector specifying
%                           color channel pre-processing. This option can
%                           be used when you have a training set that
%                           contains both color and grayscale image data
%                           and you need data created by the datastore to
%                           be strictly color or grayscale. Options are:
%                           'gray2rgb','rgb2gray','none'. For example, if
%                           you need to train a network that expects color
%                           images but some of the images in your training
%                           set are grayscale, then specifying the option
%                           'gray2rgb' will replicate the color channels of
%                           the grayscale images in the input image set to
%                           create MxNx3 output images.
%
%                           Default: 'none'
%
%   'DataAugmentation'      A scalar imageDataAugmenter object, string, or
%                           character array that specifies
%                           the kinds of image data augmentation that will
%                           be applied to generated images.
%
%                           Default: 'none'
%
%   'DispatchInBackground'  Accelerate image augmentation by asyncronously
%                           reading, augmenting, and queueing augmented
%                           images for use in training. Requires Parallel
%                           Computing Toolbox.
%
%                           Default: false
%
%   'OutputSizeMode'        A scalar string or character vector specifying the
%                           technique used to adjust image sizes to the
%                           specified 'OutputSize'. Options are: 'resize',
%                           'centercrop', 'randcrop'.
%
%                           Default: 'resize'
%
%   augmentedImageDatastore Properties:
%
%       ColorPreprocessing      - Defines color channel manipulation
%       DataAugmentation        - Defines data augmentation used
%       DispatchInBackground    - Whether background dispatch is used
%       MiniBatchSize           - Number of images returned in each read
%       NumObservations         - Total number of images in an epoch
%       OutputSize              - Vector of [width,height] of output images
%       OutputSizeMode          - Method used to resize output images
%
%   augmentedImageDatastore Methods:
%
%       augmentedImageDatastore - Construct an augmentedImageDatastore
%       hasdata                 - Returns true if there is more data in the datastore
%       partitionByIndex        - Partitions an augmentedImageDatastore given indices
%       preview                 - Reads the first image from the datastore
%       read                    - Reads a MiniBatch of data from the datastore
%       readall                 - Reads all observations from the datastore
%       readByIndex             - Random access read from datastore given indices
%       reset                   - Resets datastore to the start of the data
%       shuffle                 - Shuffles the observations in the datastore
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
%       imageInputLayer(imageSize, 'Normalization', 'none');
%       convolution2dLayer(5,20);
%       reluLayer();
%       maxPooling2dLayer(2,'Stride',2);
%       fullyConnectedLayer(10);
%       softmaxLayer();
%       classificationLayer()];
%
%   opts = trainingOptions('sgdm', 'Plots', 'training-progress');
%
%   imageAugmenter = imageDataAugmenter('RandRotation',[-10 10]);
%
%   ds = augmentedImageDatastore(imageSize,XTrain,YTrain,'DataAugmentation',imageAugmenter);
%
%   net = trainNetwork(ds, layers, opts);
%
%   Example 2
%   ---------
%   Resize all images in imageDatastore during inference to the required 
%   size of the inputImageLayer of alexnet.
%
%   net = alexnet;
%   imds = imageDatastore(fullfile(matlabroot,'toolbox','matlab','imagesci','peppers.png'));
%   imds.Files = repelem(imds.Files,5,1);
%   ds = augmentedImageDatastore(net.Layers(1).InputSize(1:2),imds);
%   Ypred = classify(net,ds)

% See also imageDataAugmenter, imageInputLayer, trainNetwork

% Copyright 2017 The MathWorks, Inc.

classdef augmentedImageDatastore <...
        matlab.io.Datastore &...
        matlab.io.datastore.MiniBatchable &...
        matlab.io.datastore.Shuffleable &...
        matlab.io.datastore.BackgroundDispatchable &...
        matlab.io.datastore.PartitionableByIndex
    
    properties (Dependent)
        %MiniBatchSize - MiniBatch Size
        %
        %   The number of observations returned as rows in the table
        %   returned by the read method.
        MiniBatchSize
    end
    
    properties (SetAccess = protected, Dependent)
        %NumObservations - Number of observations
        %
        %   The number of observations in the datastore. 
       NumObservations 
    end
       
    properties (Access = private)
        DatastoreInternal
        ImageAugmenter
    end
    
    properties (SetAccess = private)
        
        %DataAugmentation - Augmentation applied to input images
        %
        %    DataAugmentation is a scalar imageDataAugmenter object or a
        %    character vector or string. When DataAugmentation is 'none' 
        %    no augmentation is applied to input images.
        DataAugmentation
        
        %ColorPreprocessing - Pre-processing of input image color channels
        %
        %    ColorPreprocessing is a character vector or string specifying
        %    pre-proprocessing operations performed on color channels of
        %    input images. This property is used to ensure that all output
        %    images from the datastore have the number of color channels
        %    required by inputImageLayer. Valid values are
        %    'gray2rgb','rgb2gray', and 'none'. If an input images already
        %    has the desired number of color channels, no operation is
        %    performed. For example, if 'gray2rgb' is specified and an
        %    input image already has 3 channels, no operation is performed.
        ColorPreprocessing
        
        %OutputSize - Size of output images
        %
        %    OutputSize is a two element numeric vector of the form
        %    [numRows, numColumns] that specifies the size of output images
        %    returned by augmentedImageDatastore.
        OutputSize
        
        %OutputSizeMode - Method used to resize output images.
        %
        %    OutputSizeMode is a character vector or string specifying the
        %    method used to resize output images to the requested
        %    OutputSize. Valid values are 'centercrop', 'randcrop', and 
        %   'resize' (default).
        OutputSizeMode
    end
    
    properties (Access = private)
        OutputRowsColsChannels % The expected output image size [numRows, numCols, numChannels].
    end
    
    methods
        function self = augmentedImageDatastore(varargin)
            narginchk(2,inf)
            inputs = self.parseInputs(varargin{:});

            if self.NumObservations > 0
                self.ImageAugmenter = inputs.DataAugmentation;
                self.DispatchInBackground = inputs.DispatchInBackground;
                self.determineExpectedOutputSize();
            end
        end
        
        function set.MiniBatchSize(self,batchSize)
            self.DatastoreInternal.MiniBatchSize = batchSize;
        end
        
        function batchSize = get.MiniBatchSize(self)
            batchSize = self.DatastoreInternal.MiniBatchSize;
        end
        
        function numObs = get.NumObservations(self)
            numObs = self.DatastoreInternal.NumObservations;
        end
    end
    
    methods
        function reset(self)
            self.DatastoreInternal.reset();
        end
        
        function [data,info] = readByIndex(self,indices)
            [input,info] = self.DatastoreInternal.readByIndex(indices);
            input = self.applyAugmentationPipelineToBatch(input);
            [data,info] = datastoreDataToTable(input,info);
        end
        
        function [data,info] = read(self)
            
            if ~self.hasdata()
               error(message('nnet_cnn:augmentedImageDatastore:outOfData')); 
            end
            
            [input,info] = self.DatastoreInternal.read();
            input = self.applyAugmentationPipelineToBatch(input);
            [data,info] = datastoreDataToTable(input,info);
        end
        
        function newds = shuffle(self)
            dsinternal = shuffle(self.DatastoreInternal);
            newds = copy(self);
            newds.DatastoreInternal = dsinternal;
        end
                
        function TF = hasdata(self)
           TF = hasdata(self.DatastoreInternal); 
        end
        
        function newds = partitionByIndex(self,indices)
            newds = copy(self);
            newds.DatastoreInternal = partitionByIndex(self.DatastoreInternal,indices);
        end
        
    end
    
    methods (Hidden)
        
        % Reorder is an undocumented interface for doing in place shuffling
        % without a copy.
        function reorder(self,indices)
            self.DatastoreInternal.reorder(indices);
        end
        
        function frac = progress(self)
          frac = progress(self.DatastoreInternal);
        end
        
    end
    
    methods (Access = private)
        
        function determineExpectedOutputSize(self)
            
            % If a user specifies a ColorPreprocessing option, we know the
            % number of channels to expect in each mini-batch. If they
            % don't specify a ColorPreprocessing option, we need to look at
            % an example from the underlying datastore and assume all
            % images will have a consistent number of channels when forming
            % mini-batches.
            if strcmp(self.ColorPreprocessing,'rgb2gray')
                self.OutputRowsColsChannels = [self.OutputSize,1];
            elseif strcmp(self.ColorPreprocessing,'gray2rgb')
                self.OutputRowsColsChannels = [self.OutputSize,3];
            elseif strcmp(self.ColorPreprocessing,'none')
                origMiniBatchSize = self.MiniBatchSize;
                self.DatastoreInternal.MiniBatchSize = 1;
                X = self.DatastoreInternal.read();
                if iscell(X)
                    X = X{1};
                end
                self.DatastoreInternal.MiniBatchSize = origMiniBatchSize;
                self.DatastoreInternal.reset();
                exampleNumChannels = size(X,3);
                self.OutputRowsColsChannels = [self.OutputSize,exampleNumChannels];
            else
                assert(false,'Unexpected ColorPreprocessing option.');
            end
            
        end
        
        function Xout = applyAugmentationPipelineToBatch(self,X)
            if iscell(X)
                Xout = cellfun(@(c) self.applyAugmentationPipeline(c),X,'UniformOutput',false);
            else
                batchSize = size(X,4);
                Xout = cell(batchSize,1);
                for obs = 1:batchSize
                    temp = self.preprocessColor(X(:,:,:,obs));
                    temp = self.augmentData(temp);
                    Xout{obs} = self.resizeData(temp);
                end
            end
        end
        
        function Xout = applyAugmentationPipeline(self,X)
            if isequal(self.ColorPreprocessing,'none') && (size(X,3) ~= self.OutputRowsColsChannels(3))
               error(message('nnet_cnn:augmentedImageDatastore:mismatchedNumberOfChannels','''ColorPreprocessing'''));
            end
            temp = self.preprocessColor(X);
            temp = self.augmentData(temp);
            Xout = self.resizeData(temp);
        end
        
        function miniBatchData = augmentData(self,miniBatchData)
            if ~strcmp(self.DataAugmentation,'none')
                miniBatchData = self.ImageAugmenter.augment(miniBatchData);
            end
        end
        
        function Xout = resizeData(self,X)
            
            inputSize = size(X);
            if isequal(inputSize(1:2),self.OutputSize)
                Xout = X; % no-op if X is already desired Outputsize
                return
            end
            
            if strcmp(self.OutputSizeMode,'resize')
                Xout = augmentedImageDatastore.resizeImage(X,self.OutputSize);
            elseif strcmp(self.OutputSizeMode,'centercrop')
                Xout = augmentedImageDatastore.centerCrop(X,self.OutputSize);
            elseif strcmp(self.OutputSizeMode,'randcrop')
                Xout = augmentedImageDatastore.randCrop(X,self.OutputSize);
            end
        end
        
        function Xout = preprocessColor(self,X)
            
            if strcmp(self.ColorPreprocessing,'rgb2gray')
                Xout = convertRGBToGrayscale(X);
            elseif strcmp(self.ColorPreprocessing,'gray2rgb')
                Xout = convertGrayscaleToRGB(X);
            elseif strcmp(self.ColorPreprocessing,'none')
                Xout = X;
            end
        end
    end
    
    methods (Access = 'private')
        
        function inputStruct = parseInputs(self,varargin)
                        
            p = inputParser();
              
            p.addRequired('outputSize',@outputSizeValidator);
            p.addRequired('X');
            p.addOptional('Y',[]);
            p.addParameter('DataAugmentation','none',@augmentationValidator);
            
            colorPreprocessing = 'none';
            p.addParameter('ColorPreprocessing','none',@colorPreprocessingValidator);
            
            
            outputSizeMode = 'resize';
            p.addParameter('OutputSizeMode','resize',@outputSizeModeValidator);
            
            backgroundExecutionValidator = @(TF) validateattributes(TF,...
                {'numeric','logical'},{'scalar','real'},mfilename,'BackgroundExecution');
            p.addParameter('DispatchInBackground',false,backgroundExecutionValidator);
            p.addParameter('BackgroundExecution',false,backgroundExecutionValidator);
            
            responseNames = [];
            if (istable(varargin{2}) && ~isempty(varargin{2}))
                tbl = varargin{2};
                if (length(varargin) > 2) && (ischar(varargin{3}) || isstring(varargin{3}) || iscellstr(varargin{3}))
                    if checkValidResponseNames(varargin{3},tbl)
                        responseNames = varargin{3};
                        varargin(3) = [];
                    end
                end
            end
            
            p.parse(varargin{:});
            inputStruct = manageDispatchInBackgroundNameValue(p);
            
            self.DataAugmentation = inputStruct.DataAugmentation;
            self.OutputSize = inputStruct.outputSize(1:2);
            self.OutputSizeMode = outputSizeMode;
            self.ColorPreprocessing = colorPreprocessing;
                    
            % Check if Y was specified for table or imageDatastore inputs.
            propertiesWithDefaultValues = string(p.UsingDefaults);
            if (isa(inputStruct.X,'matlab.io.datastore.ImageDatastore') || isa(inputStruct.X,'table')) && ~any(propertiesWithDefaultValues == "Y")
                error(message('nnet_cnn:augmentedImageDatastore:invalidYSpecification',class(inputStruct.X)));
            end
                                    
            if ~isempty(responseNames)
                inputStruct.X = selectResponsesFromTable(inputStruct.X,responseNames);
                inputStruct.Y = responseNames;
            end
            
            % Validate numeric inputs
            if isnumeric(inputStruct.X)
                validateattributes(inputStruct.X,{'single','double','logical','uint8','int8','uint16','int16','uint32','int32'},...
                    {'nonsparse','real'},mfilename,'X');
                
                validateattributes(inputStruct.Y,{'single','double','logical','uint8','int8','uint16','int16','uint32','int32','categorical'},...
                    {'nonsparse'},mfilename,'Y');
            end
                            
            try
                self.DatastoreInternal = nnet.internal.cnn.MiniBatchDatastoreFactory.createMiniBatchDatastore(inputStruct.X,inputStruct.Y);
            catch ME
                throwAsCaller(ME);
            end
            
            function TF = colorPreprocessingValidator(sIn)
                colorPreprocessing = validatestring(sIn,{'none','rgb2gray','gray2rgb'},...
                    mfilename,'ColorPreprocessing');
                
                TF = true;
            end
            
            function TF = outputSizeModeValidator(sIn)
                outputSizeMode = validatestring(sIn,...
                    {'resize','centercrop','randcrop'},mfilename,'OutputSizeMode');
                
                TF = true;
            end
            
            function TF = outputSizeValidator(sizeIn)
               
                validateattributes(sizeIn,...
                {'numeric'},{'vector','integer','finite','nonsparse','real','positive'},mfilename,'OutputSize');
            
                if (numel(sizeIn) ~= 2) && (numel(sizeIn) ~=3)
                   error(message('nnet_cnn:augmentedImageDatastore:invalidOutputSize')); 
                end
                
                TF = true;
                
            end
            
        end
        
    end
    
    methods(Static, Hidden = true)
        function self = loadobj(S)
                        
            self = augmentedImageDatastore(S.OutputSize,S.DatasourceInternal,...
                'DispatchInBackground',S.BackgroundExecution,...
                'ColorPreprocessing',S.ColorPreprocessing,...
                'DataAugmentation',S.DataAugmentation,...
                'OutputSizeMode',S.OutputSizeMode);
            
            if isfield(S,'MiniBatchSize')
                self.MiniBatchSize = S.MiniBatchSize;
            else
                self.MiniBatchSize = 128;
            end
        end
    end
    
    methods (Hidden)
        function S = saveobj(self)
            S = struct('BackgroundExecution',self.DispatchInBackground,...
                'ColorPreprocessing',self.ColorPreprocessing,...
                'DataAugmentation',self.DataAugmentation,...
                'OutputSize',self.OutputSize,...
                'OutputSizeMode',self.OutputSizeMode,...
                'DatasourceInternal',self.DatastoreInternal,...
                'MiniBatchSize',self.MiniBatchSize);
        end
        
    end
    
    methods (Hidden, Static)
        
        
        function imOut = resizeImage(im,outputSize)
            
            ippResizeSupportedWithCast = isa(im,'int8') || isa(im,'uint16') || isa(im,'int16');
            ippResizeSupportedForType = isa(im,'uint8') || isa(im,'single');
            ippResizeSupported = ippResizeSupportedWithCast || ippResizeSupportedForType;
            
            if ippResizeSupportedWithCast
                im = single(im);
            end
            
            if ippResizeSupported
                imOut = nnet.internal.cnnhost.resizeImage2D(im,outputSize,'linear',true);
            else
                imOut = imresize(im,'OutputSize',outputSize,'method','bilinear');
            end
            
        end
        
        function im = centerCrop(im,outputSize)
            
            sizeInput = size(im);
            if any(sizeInput(1:2) < outputSize)
               error(message('nnet_cnn:augmentedImageDatastore:invalidCropOutputSize','''OutputSizeMode''',mfilename, '''centercrop''','''OutputSize''')); 
            end
            
            x = (size(im,2) - outputSize(2)) / 2;
            y = (size(im,1) - outputSize(1)) / 2;
                        
            im = augmentedImageDatastore.crop(im,...
                [x y, outputSize(2), outputSize(1)]);
        end
        
        function rect = randCropRect(im,outputSize)
            % Pick random coordinates within the image bounds
            % for the top-left corner of the cropping rectangle.
            range_x = size(im,2) - outputSize(2);
            range_y = size(im,1) - outputSize(1);
            
            x = range_x * rand;
            y = range_y * rand;
            rect = [x y outputSize(2), outputSize(1)];
        end
                     
        function im = randCrop(im,outputSize)
            sizeInput = size(im);
            if any(sizeInput(1:2) < outputSize)
                error(message('nnet_cnn:augmentedImageDatastore:invalidCropOutputSize','''OutputSizeMode''',mfilename, '''randcrop''','''OutputSize'''));
            end
            rect = augmentedImageDatastore.randCropRect(im,outputSize);
            im = augmentedImageDatastore.crop(im,rect);
        end
        
        function B = crop(A,rect)
            % rect is [x y width height] in floating point.
            % Convert from (x,y) real coordinates to [m,n] indices.
            rect = floor(rect);
            
            m1 = rect(2) + 1;
            m2 = rect(2) + rect(4);
            
            n1 = rect(1) + 1;
            n2 = rect(1) + rect(3);
                        
            m1 = min(size(A,1),max(1,m1));
            m2 = min(size(A,1),max(1,m2));
            n1 = min(size(A,2),max(1,n1));
            n2 = min(size(A,2),max(1,n2));
            
            B = A(m1:m2, n1:n2, :, :);
        end
    end
end

function TF = checkValidResponseNames(responseNames, tbl)
% iAssertValidResponseNames   Assert that the response names are variables
% of the table and they do not refer to the first column.
variableNames = tbl.Properties.VariableNames;
refersToFirstColumn = ismember( variableNames(1), responseNames );
responseNamesAreAllVariables = all( ismember(responseNames,variableNames) );
TF = ~(refersToFirstColumn || ~responseNamesAreAllVariables);
end

function resTbl = selectResponsesFromTable(tbl, responseNames)
% iSelectResponsesFromTable   Return a new table with only the first column
% (predictors) and the variables specified in responseNames.
variableNames = tbl.Properties.VariableNames;
varTF = ismember(variableNames, responseNames);
% Make sure to select predictors (first column) as well
varTF(1) = 1;
resTbl = tbl(:,varTF);
end

function TF = augmentationValidator(valIn)

if ischar(valIn) || isstring(valIn)
    TF = string('none').contains(lower(valIn)); %#ok<STRQUOT>
elseif isa(valIn,'imageDataAugmenter') && isscalar(valIn)
    TF = true;
else
    TF = false;
end

end

function im = convertRGBToGrayscale(im)
if (ndims(im) == 3 && size(im,3) == 3)
    im = rgb2gray(im);
end
end

function im = convertGrayscaleToRGB(im)
if size(im,3) == 1
    im = repmat(im,[1 1 3]);
end
end

function c = convert4DArrayToCell(X)

if isnumeric(X)
    c = cell([size(X,4),1]);
    for i = 1:length(c)
        c{i} = X(:,:,:,i);
    end
else
   c = X; 
end

end

function [data,info] = datastoreDataToTable(input,info)

response = info.Response;
info = rmfield(info,'Response');
if isempty(response)
    data = table(input);
else
    response = convert4DArrayToCell(response);
    data = table(input,response);
end

end

function resultsStruct = manageDispatchInBackgroundNameValue(p)

resultsStruct = p.Results;

DispatchInBackgroundSpecified = ~any(strncmp('DispatchInBackground',p.UsingDefaults,length('DispatchInBackground')));
BackgroundExecutionSpecified = ~any(strncmp('BackgroundExecution',p.UsingDefaults,length('BackgroundExecution')));

% In R2017b, BackgroundExecution was name used to control
% DispatchInBackground. Allow either to be specified.
if BackgroundExecutionSpecified && ~DispatchInBackgroundSpecified
    resultsStruct.DispatchInBackground = resultsStruct.BackgroundExecution;
end

end

