classdef MiniBatchDatastoreDispatcher < nnet.internal.cnn.DataDispatcher &...
                                        nnet.internal.cnn.BackgroundCapableDispatcher &...
                                        nnet.internal.cnn.DistributableMiniBatchDatastoreDispatcher
    
    % MiniBatchDatastoreDispatcher class to dispatch 4D data one mini batch at a
    %   time from a MiniBatchDatastore
    %
    % Input data    - MiniBatchDatastore.
    %
    % Output data   - 4D data where the last dimension is the number of
    %               observations in that mini batch. The type of the data
    %               in output will be the same as the one in input
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % ImageSize  (1x3 int) Size of each image to be dispatched
        ImageSize
        
        % ResponseSize   (1x3 int) Size of each response to be dispatched
        ResponseSize
        
        % IsDone (logical)     True if there is no more data to dispatch
        IsDone
        
        % NumObservations (int) Number of observations in the data set
        NumObservations
        
        % Categories (categorical array) Array of categories corresponding 
        %            to training data labels. It holds also ordinality of 
        %            the labels.
        Categories = categorical();
        
        % ResponseNames (cellstr) Array of response names corresponding to
        %               training data response names. Since we cannot get
        %               the names anywhere for an array, we will use a
        %               fixed response name.
        ResponseNames = {'Response'};
        
    end
    
    properties(Dependent)
        % MiniBatchSize (int)   Number of elements in a mini batch
        MiniBatchSize
    end
    
    properties(SetAccess = private, Dependent)
        % ClassNames (cellstr) Array of class names corresponding to
        %            training data labels.
        ClassNames;
    end
    
    properties
        % Precision   Precision used for dispatched data
        Precision
        
        % EndOfEpoch    Strategy to choose how to cope with a number of
        % observation that is not divisible by the desired number of mini
        % batches
        %
        % Allowed values: 'truncateLast', 'discardLast'
        EndOfEpoch
    end
    
    properties (Access = ?nnet.internal.cnn.DistributableDispatcher)
        % Datastore  The MiniBatchDatastore we are going to
        % read data and responses from
        Datastore
        
        batchDataStore
    end
    
    properties (Access = private)
        
        % CurrentStartIndex  (int)   Current index of image to be dispatched
        CurrentStartIndex
        
        InitialMiniBatchSize % Initial mini-batch size
        
        % Whether a MiniBatchDatastore defines responses
        HasResponses
        
    end
    
    methods
        function this = MiniBatchDatastoreDispatcher(datastore, miniBatchSize, endOfEpoch, precision)
            % FourDArrayDispatcher   Constructor for 4-D array data dispatcher
            %
            % datastore         - MiniBatchDatastore object
            %
            % miniBatchSize     - Size of a mini batch express in number of
            %                   examples
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
            
            this.EndOfEpoch = endOfEpoch;
            this.Precision = precision;
            this.IsDone = false;
            
            this.Datastore = datastore;

            if ~iDatastoreContainsData(datastore)
                this.NumObservations = 0;
            else
                % Count the number of observations
                this.NumObservations = this.Datastore.NumObservations;
                
                [this.MiniBatchSize,this.InitialMiniBatchSize] = deal(miniBatchSize);
                
                % Get example
                [exampleImage,exampleResponse,this.Categories,this.ResponseNames] = getExampleInfoFromDatastore(this);
                this.HasResponses = ~isempty(exampleResponse);
                
                % Set the expected image size
                this.ImageSize = iImageSize( exampleImage );
                
                % Set the expected response size to be dispatched
                this.ResponseSize = iResponseSize(exampleResponse);
                                             
                % Run in background if a Datastore subscribes to
                % BackgroundDispatchable and UseParallel is
                % set to true.
                if isa(this.Datastore,'matlab.io.datastore.BackgroundDispatchable')
                    this.setRunInBackground(this.Datastore.DispatchInBackground);
                    this.RunInBackgroundOnAuto = false;
                end
                
            end
        end
        
        
        function [miniBatchData, miniBatchResponse, miniBatchIndices] = next(this)
            % next Get the data and response for the next mini batch and
            % correspondent indices
            
            % Return next batch of data
            if ~this.IsDone
                [miniBatchData, miniBatchResponse] = readData(this);
                iCheckCurrentInputSize(miniBatchData,this.ImageSize);
                miniBatchIndices = this.currentIndices();
            else
                miniBatchIndices = [];
                [miniBatchData,miniBatchResponse] = deal(this.Precision.cast([]));
            end
            
            if isempty(miniBatchIndices)
                return
            end
            
            this.advanceCurrentStartIndex(this.MiniBatchSize);
            
            nextMiniBatchSize = this.nextMiniBatchSize(miniBatchIndices(end)); % Manage discard vs. truncate batch behavior.
            if nextMiniBatchSize > 0
                this.MiniBatchSize = nextMiniBatchSize;
            else
                this.IsDone = true;
            end
            
        end
        
%         //RETRIEVE CLEAN DATA FOR GSGD//
        function [miniBatchData, miniBatchResponse] = captiveData(this, idx)
            % next   Get the data and response for the next mini batch and
            % correspondent indices
            start_indices = this.currentIndices();
            original_StartIndex = this.CurrentStartIndex;
            this.CurrentStartIndex = idx;
            
            % Return next batch of data
            if ~this.IsDone
                [miniBatchData, miniBatchResponse] = readCleanData(this);
                iCheckCurrentInputSize(miniBatchData,this.ImageSize);
                this.CurrentStartIndex = original_StartIndex;
                miniBatchIndices = start_indices;
            else
                miniBatchIndices = [];
                [miniBatchData,miniBatchResponse] = deal(collection.Precision.cast([]));
            end
            
            if isempty(miniBatchIndices)
                return
            end
            
            this.advanceCurrentStartIndex(0);
            
            nextMiniBatchSize = this.nextMiniBatchSize(miniBatchIndices(end)); % Manage discard vs. truncate batch behavior.
            if nextMiniBatchSize > 0
                this.MiniBatchSize = nextMiniBatchSize;
            else
                this.IsDone = true;
            end
            
        end
        
        function [miniBatchData, miniBatchResponse, miniBatchIndices] = getObservations(this,miniBatchIndices)
            
            if (this.NumObservations == 0) || isempty(miniBatchIndices)
                [miniBatchData, miniBatchResponse] = deal([], []);
                return;
            end
            
            [data,info] = this.Datastore.readByIndex(miniBatchIndices);
            
            if istable(data)
                miniBatchData = this.readObservations(data{:,1});
            else
                miniBatchData = this.readObservations(data);
            end
            
            iCheckCurrentInputSize(miniBatchData,this.ImageSize);

            if this.HasResponses
                if istable(data)
                    miniBatchResponse = this.readResponses(data{:,2});
                else
                    miniBatchResponse = this.readResponses(info.Response);
                end
            else
                miniBatchResponse = [];
            end
            
        end
                        
        function start(this)
            % start     Go to first mini batch
            this.IsDone = false;
            this.MiniBatchSize = this.InitialMiniBatchSize;
            if (this.NumObservations ~= 0)
                this.Datastore.reset();
            end
            this.CurrentStartIndex = 1;
        end
        
        function shuffle(this)
            % shuffle   Shuffle the data
            if (this.NumObservations ~= 0)
                this.Datastore = this.Datastore.shuffle();
            end
        end
        
        function reorder(this,indices)
           % reorder  Shuffle the data in a specific order
           %
           % Note, this will only be called if underlying Datastore implements
           % reorder, as checked in implementsReorder().
           if (this.NumObservations ~= 0)
               this.checkValidReorderIndices(indices);
               this.Datastore.reorder(indices);
           end
        end
        
        function set.MiniBatchSize(this, value)
            if (this.NumObservations ~= 0)
                value = min(value, this.NumObservations);
                this.Datastore.MiniBatchSize = value;
            end
        end
        
        function batchSize = get.MiniBatchSize(this)
            if (this.NumObservations ~= 0)
                batchSize = this.Datastore.MiniBatchSize;
            else
                batchSize = 0;
            end
        end
        
        function names = get.ClassNames(this)
            names = categories(this.Categories);
            names = cellstr(names);
        end

        function TF = implementsReorder(this)
            % implementsReorder  Whether a dispatcher implements the reorder interface.
            % In this particular dispatcher, we need to interrogate the underlying MiniBatchDatastore.
            % This is an overload of the definition in
            % BackgroundCapableDispatcher.
            
            TF = ismethod(this.Datastore,'reorder');
            
        end
                
    end
        
    methods (Access = private)
        
        function [miniBatchData, miniBatchResponse] = readCleanData(this)
            % readData  Read data and response corresponding to indices
            [data,info] = read(this.Datastore);
            if istable(data)
                X = data{:,1};
            else
                X = data;
            end
            miniBatchData = this.readObservations(X);
            if this.HasResponses
                if istable(data)
                    Y = data{:,2};
                else
                    Y = info.Response;
                end
                miniBatchResponse = this.readResponses(Y);
            else
                miniBatchResponse = [];
            end
        end
        function [miniBatchData, miniBatchResponse] = readData(this)
            % readData  Read data and response corresponding to indices
            [data,info] = read(this.Datastore);
            if istable(data)
                X = data{:,1};
            else
                X = data;
            end
            miniBatchData = this.readObservations(X);
            if this.HasResponses
                if istable(data)
                    Y = data{:,2};
                else
                    Y = info.Response;
                end
                miniBatchResponse = this.readResponses(Y);
            else
                miniBatchResponse = [];
            end
        end
        
        function observations = readObservations(this,X)
            if isempty(X)
                observations = [];
            else
                observations = iCellTo4DArray(X);
            end
            observations = this.Precision.cast(observations);
        end
        
        function responses = readResponses(this,Y)
            if isempty(Y)
                responses = [];
            else
                if iscell(Y)
                    Y = iCellTo4DArray(Y);
                end
                if iscategorical(Y)
                    % Categorical vector of responses
                    responses = iDummify( Y );                
                else
                    responses = Y;
                end
            end
            % Cast to the right precision
            responses = this.Precision.cast( responses );
        end
        
    end
    
    methods (Access = private)
        
        function [exampleInput,exampleResponse,categories,responseNames] = getExampleInfoFromDatastore(this)
            % getExampleInfoFromDatastore   Extract examples of input,
            % response, and classnames from a Datastore.
            
            this.MiniBatchSize = 1;
            
            [data,info] = read(this.Datastore);

            if isa(this.Datastore,'matlab.io.datastore.internal.ResponseNameable')
                responseNames = this.Datastore.ResponseNames;
            else
                responseNames = iGetResponseNames(data);
            end
                            
            if istable(data)
                X = data{:,1};
            else
                X = data;
            end
            exampleInput = this.readObservations(X);
            exampleInput = exampleInput(:,:,:,1);

            hasResponses = (istable(data) && (size(data,2) > 1)) || (isfield(info,'Response') && ~isempty(info.Response));
            if hasResponses
                if istable(data)
                    Y = data{:,2};
                else
                    Y = info.Response;
                end
                exampleResponse = this.readResponses(Y);
                exampleResponse = exampleResponse(:,:,:,1);
                
                categories = iGetCategories(Y);
            else
               exampleResponse = [];
               categories = categorical();
            end
            
            this.MiniBatchSize = this.InitialMiniBatchSize;
            this.Datastore.reset();
            
        end
        
        function indices = currentIndices(this)
            indices = this.CurrentStartIndex : (this.CurrentStartIndex + this.MiniBatchSize - 1);
        end
        
        function [oldIdx, newIdx] = advanceCurrentStartIndex( this, n )
            % advanceCurrentStartIndex     Advance current index of n positions and return
            % its old and new value
            oldIdx = this.CurrentStartIndex;
            this.CurrentStartIndex = this.CurrentStartIndex + n;
            newIdx = this.CurrentStartIndex;
        end
        
        function miniBatchSize = nextMiniBatchSize( this, currentEndIdx )
            % nextMiniBatchSize   Compute the size of the next mini batch
                        
            miniBatchSize = min( this.MiniBatchSize, this.NumObservations - currentEndIdx );
            
            if isequal(this.EndOfEpoch, 'discardLast') && miniBatchSize<this.MiniBatchSize
                miniBatchSize = 0;
            end
        end
        
    end
    
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

function TF = iDatastoreContainsData( ds )
    TF = ~(isempty(ds) || (ds.NumObservations == 0));
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(errorID, getString(message(errorID, varargin{:})));
end

function throwVariableSizesException(e)
% throwVariableSizesException   Throws a subsassigndimmismatch exception as
% a VariableImageSizes exception
if (strcmp(e.identifier,'MATLAB:catenate:dimensionMismatch'))
    exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:MiniBatchDatastoreDispatcher:VariableImageSizes');
    throwAsCaller(exception)
else
    rethrow(e)
end
end

function iCheckCurrentInputSize(X,expectedSize)

if ~isequal(expectedSize,iImageSize(X(:,:,:,1)))
   e =  MException('nnet_cnn:internal:cnn:MiniBatchDatastoreDispatcher:VariableImageSizes',...
       getString(message('nnet_cnn:internal:cnn:MiniBatchDatastoreDispatcher:VariableImageSizes')));
   throwAsCaller(e);
end

end

function dummy = iDummify(categoricalIn)
% iDummify   Dummify a categorical vector of size numObservations x 1 to
% return a 4-D array of size 1 x 1 x numClasses x numObservations
if isempty( categoricalIn )
    numClasses = 0;
    numObs = 1;
    dummy = zeros( 1, 1, numClasses, numObs );
else
    if isvector(categoricalIn)
        dummy = nnet.internal.cnn.util.dummify(categoricalIn);
    else
        dummy = iDummify4dArray(categoricalIn);
    end
end
end

function dummy = iDummify4dArray(C)
numCategories = numel(categories(C));
[H, W, ~, numObservations] = size(C);
dummifiedSize = [H, W, numCategories, numObservations];
dummy = zeros(dummifiedSize, 'single');
C = iMakeVertical( C );

[X,Y,Z] = meshgrid(1:W, 1:H, 1:numObservations);

X = iMakeVertical(X);
Y = iMakeVertical(Y);
Z = iMakeVertical(Z);

% Remove missing labels. These are pixels we should ignore during
% training. The dummified output is all zeros along the 3rd dims and are
% ignored during the loss computation.
[C, removed] = rmmissing(C);
X(removed) = [];
Y(removed) = [];
Z(removed) = [];

idx = sub2ind(dummifiedSize, Y(:), X(:), int32(C), Z(:));
dummy(idx) = 1;
end

function cats = iGetCategories(response)

if iscell(response)
    response = response{1};    
end
cats = nnet.internal.cnn.util.categoriesFromResponse(response);
end

function imageSize = iImageSize(image)
% iImageSize    Retrieve the image size of an image, adding a third
% dimension if grayscale
[ imageSize{1:3} ] = size(image);
imageSize = cell2mat( imageSize );
end

function responseSize = iResponseSize(response)
% iResponseSize   Return the size of the response in the first three
% dimensions.
[responseSize(1), responseSize(2), responseSize(3), ~] = size(response);
end

function vec = iMakeVertical( vec )
    vec = reshape( vec, numel( vec ), 1 );
end

function responseNames = iGetResponseNames( Data )

if istable(Data)
    responseNames = Data.Properties.VariableNames(2:end);
    % To be consistent with ClassNames, return a column array
    responseNames = responseNames';
else
    % Use default response name
    responseNames = {'Response'};
end
end
