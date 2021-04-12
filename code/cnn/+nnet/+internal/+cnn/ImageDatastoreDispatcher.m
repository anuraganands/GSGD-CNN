classdef ImageDatastoreDispatcher < ...
        nnet.internal.cnn.DataDispatcher & ...
        nnet.internal.cnn.DistributableImageDatastoreDispatcher & ...
        nnet.internal.cnn.BackgroundCapableDispatcher
    % ImageDatastoreDispatcher  class to dispatch 4D data from
    %   ImageDatastore data
    %
    % Input data    - an image datastore containing either RGB or grayscale
    %               images of the same size
    % Output data   - 4D data where the fourth dimension is the number of
    %               observations in that mini batch. If the input is a
    %               grayscale image then the 3rd dimension will be 1
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % ImageSize  (1x3 int) Size of each image to be dispatched
        ImageSize
        
        % ResponseSize   (1x3 int) Size of each response to be dispatched
        ResponseSize
        
        % NumObservations (int) Number of observations in the data set
        NumObservations
        
        % ClassNames (cellstr) Array of class names corresponding to
        %            training data labels.
        ClassNames
        
        % ResponseNames (cellstr) Array of response names corresponding to
        %               training data response names. Since we cannot get
        %               the names anywhere for an array, we will use a
        %               fixed response name.
        ResponseNames = {'Response'};
        
        % PrivateMiniBatchSize (int)   Number of elements in a mini batch
        PrivateMiniBatchSize
    end
    
    properties(SetAccess = public)
        % EndOfEpoch    Strategy for how to cope with the last mini-batch when the number
        % of observations is not divisible by the number of mini batches.
        %
        % Allowed values: 'truncateLast', 'discardLast'
        EndOfEpoch
        
        % Precision Precision used for dispatched data
        Precision
    end
    
    properties (SetAccess = private, Dependent)
        % IsDone (logical)     True if there is no more data to dispatch
        IsDone
    end
    
    properties (Access = ?nnet.internal.cnn.DistributableDispatcher)
        % Datastore  (ImageDatastore)    The ImageDatastore we are going to
        % read data and responses from
        Datastore
    end
    
    properties (Access = private)
        % CurrentIndex  (int)   Current index of image to be dispatched
        CurrentIndex
        
        % ExampleImage    An example of the images in the dataset
        ExampleImage
        
        % ExampleLabel     An example categorical label from the dataset
        ExampleLabel
    end
    
    properties(Dependent)
        % MiniBatchSize (int)   Number of elements in a mini batch
        MiniBatchSize
    end
    
    methods
        function this = ImageDatastoreDispatcher(imageDatastore, miniBatchSize, endOfEpoch, precision)
            % ImageDatastoreDispatcher   Constructor for array data dispatcher
            %
            % imageDatastore    - An ImageDatastore containing the images
            % miniBatchSize     - Size of a mini batch express in number of
            %                   images
            % endOfEpoch        - Strategy to choose how to cope with a
            %                   number of observation that is not divisible
            %                   by the desired number of mini batches
            %                   One of:
            %                   'truncateLast' to truncate the last mini
            %                   batch
            %                   'discardLast' to discard the last mini
            %                   batch
            % precision         - What precision to use for the dispatched
            %                   data
            
            backend = iGetBackEnd(imageDatastore);
            
            if backend.Name ~= "local"
                imageDatastore = iOptimizeRemotePrefetch(imageDatastore, miniBatchSize, backend);
            end
            
            this.Datastore = imageDatastore;            
            assert(isequal(endOfEpoch,'truncateLast') || isequal(endOfEpoch,'discardLast'), 'nnet.internal.cnn.ImageDatastoreDispatcher error: endOfEpoch should be one of ''truncateLast'', ''discardLast''.');
            this.EndOfEpoch = endOfEpoch;
            this.Precision = precision;
            
            % Need to allow empty datastores
            if isempty( imageDatastore )
                this.NumObservations = 0;
                this.MiniBatchSize = 0;
                % Other properties can remain unset as they won't be used
                % in this case
                return;
            end
            
            % Count the number of observations
            this.NumObservations = numel( imageDatastore.Files );
            
            this.MiniBatchSize = miniBatchSize;
            
            % Read an example image
            [this.ExampleImage, exampleInfo] = imageDatastore.readimage(1);
            
            % Record an example label to capture type
            this.ExampleLabel = exampleInfo.Label;
            
            % Set the expected image size
            this.ImageSize = iImageSize( this.ExampleImage );
            
            % Set the expected response size to be dispatched
            this.ResponseSize = iResponseSize( iDummify( this.ExampleLabel, 1 ) );
            
            % Get class names from labels
            this.ClassNames = iGetClassNames(this.Datastore);
            
            % Only automatically run in background if the datastore has a
            % custom ReadFcn, otherwise it will likely be slower than the
            % threaded prefetch used automatically
            if ~nnet.internal.cnn.util.imdsHasCustomReadFcn(this.Datastore)
                this.RunInBackgroundOnAuto = false;
            end
            
        end
        
        function tf = get.IsDone( this )
            if isempty(this.Datastore)
                tf = true;
            elseif isequal(this.EndOfEpoch, 'truncateLast')
                tf = this.CurrentIndex >= this.NumObservations;
            else % discardLast
                % If the current index plus the mini-batch size takes us
                % beyond the end of the number of observations, then we are
                % done
                tf = this.CurrentIndex + this.MiniBatchSize > this.NumObservations;
            end
        end
        
        function [miniBatchData, miniBatchResponse, miniBatchIndices] = next(this)
            % next   Get the data and response for the next mini batch and
            % correspondent indices
            currentMiniBatchSize = this.nextMiniBatchSize();
            if currentMiniBatchSize>0
                miniBatchLabels = repmat( this.ExampleLabel, currentMiniBatchSize, 1 );
                this.Datastore.ReadSize = currentMiniBatchSize;
                [images, info] = this.Datastore.read();
                
                miniBatchData = iCellTo4DArray( images );
                
                % If there are no categories don't try to record empty
                % labels
                if ~isempty(info.Label)
                    miniBatchLabels = info.Label;
                end
                miniBatchIndices = this.nextIndices( currentMiniBatchSize );
                miniBatchResponse = iDummify( miniBatchLabels, currentMiniBatchSize );
            else
                miniBatchData = [];
                miniBatchResponse = [];
                miniBatchIndices = [];
            end
            
            % Convert to correct types
            miniBatchData = this.Precision.cast( miniBatchData );
            miniBatchResponse = this.Precision.cast( miniBatchResponse );
        end
         
        function [data, response, indices] = getObservations(this, indices)
        % getObservations  Overload of method to retrieve specific
        % observations. The implementation for image datastore is
        % inefficient so should only be used when cost of dispatch is
        % masked (because it happens in the background for instance).
        
            n = numel(indices);
            if n == 0
                data = [];
                response = [];
            else
                % Index the labels
                if isempty(this.Datastore.Labels)
                    labels = repmat( this.ExampleLabel, n, 1 );
                else
                    labels = this.Datastore.Labels(indices);
                end

                % Create datastore partition via a copy and index. This is
                % faster than constructing a new datastore with the new
                % files.
                subds = copy( this.Datastore );
                subds.Files = this.Datastore.Files(indices);
                
                % Read and concatenate
                data = iCellTo4DArray( subds.readall() );
                data = this.Precision.cast( data );
                
                % Get the response from the labels
                response = iDummify( labels, n );
                response = this.Precision.cast( response );
            end
        end
       
        function start(this)
            % start     Set the next the mini batch to be the first mini
            % batch in the epoch
            if ~isempty(this.Datastore)
                reset( this.Datastore );
            end
            this.CurrentIndex = 0;
        end
        
        function shuffle(this)
            % shuffle   Shuffle the data
            if ~isempty(this.Datastore)
                this.Datastore = this.Datastore.shuffle();
            end
        end
        
        function reorder(this, indices)
            % reorder   Shuffle data to a specific order
            if ~isempty(this.Datastore)
                this.checkValidReorderIndices(indices);
                newDatastore = copy(this.Datastore);
                newDatastore.Files = this.Datastore.Files(indices);
                newDatastore.Labels = this.Datastore.Labels(indices);
                this.Datastore = newDatastore;
            end
        end
        
        function value = get.MiniBatchSize(this)
            value = this.PrivateMiniBatchSize;
        end
        
        function set.MiniBatchSize(this, value)
            value = min(value, this.NumObservations);
            this.PrivateMiniBatchSize = value;
        end
    end
    
    methods (Access = private)
        function indices = nextIndices( this, n )
            % nextIndices   Return the next n indices
            [startIdx, endIdx] = this.advanceCurrentIndex( n );
            indices = startIdx+1:endIdx;
            % The returned indices are expected to be a column vector
            indices = indices';
        end
        
        function [oldIdx, newIdx] = advanceCurrentIndex( this, n )
            % nextIndex     Advance current index of n positions and return
            % its old and new value
            oldIdx = this.CurrentIndex;
            this.CurrentIndex = this.CurrentIndex + n;
            newIdx = this.CurrentIndex;
        end
        
        function miniBatchSize = nextMiniBatchSize( this )
            % nextMiniBatchSize   Compute the size of the next mini batch
            miniBatchSize = min( this.MiniBatchSize, this.NumObservations - this.CurrentIndex );
            
            if isequal(this.EndOfEpoch, 'discardLast') && miniBatchSize<this.MiniBatchSize
                miniBatchSize = 0;
            end
        end
    end
end

function imageSize = iImageSize(image)
% iImageSize    Retrieve the image size of an image, adding a third
% dimension if grayscale
[ imageSize{1:3} ] = size(image);
imageSize = cell2mat( imageSize );
end

function data = iCellTo4DArray( images )
% iCellTo4DArray   Convert a cell array of images to a 4-D array. If the
% input images is already an array just return it.
if iscell( images )
    try
        data = cat(4, images{:});
    catch e
        throwVariableSizesException(e);
    end
else
    data = images;
end
end

function classnames = iGetClassNames(imds)
if isa(imds.Labels, 'categorical')
    classnames = categories( imds.Labels );
else
    classnames = {};
end
end

function dummy = iDummify(categoricalIn, numObservations)
if isempty(categoricalIn)
    numClasses = 0;
    dummy = zeros(1, 1, numClasses, numObservations);
else
    dummy = nnet.internal.cnn.util.dummify(categoricalIn);
end
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(errorID, getString(message(errorID, varargin{:})));
end

function throwVariableSizesException(e)
% throwVariableSizesException   Throws a subsassigndimmismatch exception as
% a VariableImageSizes exception
if (strcmp(e.identifier,'MATLAB:catenate:dimensionMismatch'))
    exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:ImageDatastoreDispatcher:VariableImageSizes');
    throwAsCaller(exception)
else
    rethrow(e)
end
end

function responseSize = iResponseSize(response)
% iResponseSize   Return the size of the response in the first three
% dimensions.
[responseSize(1), responseSize(2), responseSize(3), ~] = size(response);
end

function be = iGetBackEnd(imds)
% Establish if the back-end used remote or local. In the case of a remote
% back-end, it will set all the appropriate fields for the remote back-end,
% like the latency and the bandwidth

if ~isempty(imds) && all(startsWith(imds.Files(:), 's3:'))
    % For the S3 endpoint, we manually measured and gathered bandwidth and 
    % latency
    be.Name = 's3';
    be.Bandwidth = 100e6;
    be.AvgDelay = 100e-3;
else
    be.Name = 'local';
end

end

function n = iGetNumberOfLabs
% Retrieve the number of workers. If numlabs does not exist (i.e., PCT is
% not installed), simply return 1.

if exist('numlabs', 'builtin') == 5
    n = numlabs;
else
    n = 1;
end

end

function imds = iOptimizeRemotePrefetch(imds, batchSize, backEnd)
% iOptimizeRemotePrefetch Set a possibly larger number of threads and
% prefetch size in order to improve performance for a remote end-point:
%    - MaxThreads will be a multiple of the number of physical
%      cores (oversubscription)
%    - PrefetchSize is dependent on the bandwidth and on the
%      maximal delay (we will try to fully exploit the backend bandwidth)

% Get the number of available cores
ncpu = feature('numcores');

% Get the number of labs to scale down memory and threads
nlabs = iGetNumberOfLabs;

% The number of threads is set to be a multiple of the number
% of physical cores in order to hide the latency.
imds.MaxThreads  = 10 * ceil(ncpu / nlabs);

% Let's calculate the BDP = 2 * RTT * BW. The BDP represents
% the optimal number of bytes in flight during the download
BandwidthDelayProduct = 2 * backEnd.AvgDelay * backEnd.Bandwidth / nlabs;

% The number of files will be the BDP/fileSize
optimalSize = ceil(BandwidthDelayProduct / getAverageFileSize(imds));

% Make the prefetchSize a multiple of the ReadSize, in this way we don't
% risk downloading less than PrefetchSize files
imds.PrefetchSize = ceil(optimalSize/batchSize) * batchSize;

end
