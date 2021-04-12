classdef FourDArrayDispatcher < ...
        nnet.internal.cnn.DataDispatcher & ...
        nnet.internal.cnn.DistributableFourDArrayDispatcher & ...
        nnet.internal.cnn.BackgroundCapableDispatcher
    % FourDArrayDispatcher class to dispatch 4D data one mini batch at a
    %   time from 4D numeric data
    %
    % Input data    - 4D data where the last dimension is the number of
    %               observations.
    % Output data   - 4D data where the last dimension is the number of
    %               observations in that mini batch. The type of the data
    %               in output will be the same as the one in input
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
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
        
        % PrivateMiniBatchSize (int)   Number of elements in a mini batch
        PrivateMiniBatchSize
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
        
        % Data  (array)     A copy of the data in the workspace. This is a
        % 4-D array where last dimension indicates the number of examples
        Data
        
        % Response   A copy of the response data in the workspace.
        %
        % This could be:
        % - numObservations x 1 categorical vector
        % - numObservations x numResponses numeric matrix
        % - H x W x C x numObservations numeric tensor
        Response
        
    end
    
    properties (Access = ?nnet.internal.cnn.DistributableDispatcher)
        % StartIndexOfCurrentMiniBatch (int) Start index of current mini
        % batch
        StartIndexOfCurrentMiniBatch
        
        % EndIndexOfCurrentMiniBatch (int) End index of current mini batch
        EndIndexOfCurrentMiniBatch
        
        % OrderedIndices   Order to follow when indexing into the data.
        % This can keep a shuffled version of the indices.
        OrderedIndices
    end
    
    methods
        function this = FourDArrayDispatcher(data, response, miniBatchSize, endOfEpoch, precision)
            % FourDArrayDispatcher   Constructor for 4-D array data dispatcher
            %
            % data              - 4D array from the workspace where the last
            %                   dimension is the number of observations
            % response          - Data responses in the form of:
            %                   numObservations x 1 categorical vector
            %                   numObservations x numResponses numeric matrix
            %                   H x W x C x numObservations numeric tensor
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
            this.Data = data;
            this.EndOfEpoch = endOfEpoch;
            this.Precision = precision;
            
            if isempty( data )
                this.NumObservations = 0;
                this.MiniBatchSize = 0;
            else
                this.Response = response;
                this.Categories = iGetCategories(this.Response);
                this.ImageSize = iGetImageDimensionsFromArray(data);
                
                this.ResponseSize = iResponseSize(this.readResponses(1));
                this.NumObservations = size(data, 4);
                this.MiniBatchSize = miniBatchSize;
                assert(isequal(endOfEpoch,'truncateLast') || isequal(endOfEpoch,'discardLast'), 'nnet.internal.cnn.FourDArrayDispatcher error: endOfEpoch should be one of ''truncateLast'', ''discardLast''.');
                
                this.OrderedIndices = 1:this.NumObservations;
            end
            
            % It is never faster to run an in-memory dispatcher in the
            % background because indexing is faster than the communication
            % cost
            this.RunInBackgroundOnAuto = false;
        end
        
        function [miniBatchData, miniBatchResponse, miniBatchIndices] = next(this)
            % next   Get the data and response for the next mini batch and
            % correspondent indices
            
            % Map the indices into data
            miniBatchIndices = this.computeDataIndices();
            
            % Read the data
            [miniBatchData, miniBatchResponse] = this.readData(miniBatchIndices);
            
            % Advance indices of current mini batch
            this.advanceCurrentMiniBatchIndices();
        end
        
        function [data, response, indices] = getObservations(this, indices)
        % getObservations  Overload of method to retrieve specific
        % observations
            indices = this.OrderedIndices(indices);
            [data, response] = this.readData( indices );
        end
        
        function start(this)
            % start     Go to first mini batch
            this.IsDone = false;
            this.StartIndexOfCurrentMiniBatch = 1;
            
            this.EndIndexOfCurrentMiniBatch = this.MiniBatchSize;
        end
        
        function shuffle(this)
            % shuffle   Shuffle the data
            this.OrderedIndices = randperm(this.NumObservations);
        end
             
        function reorder(this, indices)
            % reorder   Shuffle the data to a specific order
            this.checkValidReorderIndices(indices);
            this.OrderedIndices = indices;
        end
        
        function value = get.MiniBatchSize(this)
            value = this.PrivateMiniBatchSize;
        end
        
        function set.MiniBatchSize(this, value)
            value = min(value, this.NumObservations);
            this.PrivateMiniBatchSize = value;
        end
        
        function names = get.ClassNames(this)
            names = categories(this.Categories);
            names = cellstr(names);
        end
    end
    
    methods (Access = private)
        function advanceCurrentMiniBatchIndices(this)
            % advanceCurrentMiniBatchIndices   Move forward start and end
            % index of current mini batch based on mini batch size
            if this.EndIndexOfCurrentMiniBatch == this.NumObservations
                % We are at the end of a cycle
                this.IsDone = true;
            elseif this.EndIndexOfCurrentMiniBatch + this.MiniBatchSize > this.NumObservations
                % Last mini batch is smaller
                if isequal(this.EndOfEpoch, 'truncateLast')
                    this.StartIndexOfCurrentMiniBatch = this.StartIndexOfCurrentMiniBatch + this.MiniBatchSize;
                    this.EndIndexOfCurrentMiniBatch = this.NumObservations;
                else % discardLast
                    % Move the starting index after the end, so that the
                    % dispatcher will return empty data
                    this.StartIndexOfCurrentMiniBatch = this.EndIndexOfCurrentMiniBatch+1;
                    this.IsDone = true;
                end
            else
                % We are in the middle of a cycle
                this.StartIndexOfCurrentMiniBatch = this.StartIndexOfCurrentMiniBatch + this.MiniBatchSize;
                this.EndIndexOfCurrentMiniBatch = this.EndIndexOfCurrentMiniBatch + this.MiniBatchSize;
            end
        end
        
        function [miniBatchData, miniBatchResponse] = readData(this, indices)
            % readData  Read data and response corresponding to indices
            miniBatchData = this.Precision.cast( this.Data(:,:,:,indices) );
            miniBatchResponse = this.readResponses( indices );
        end
        
        function responses = readResponses(this, indices)
            if isempty(this.Response)
                responses = [];
            else
                if iscategorical(this.Response)
                    % Categorical vector of responses
                    responses = iDummify( this.Response(indices) );
                elseif ismatrix(this.Response)
                    % Matrix of responses
                    responses = iMatrix2Tensor( this.Response(indices,:) );
                else
                    % 4D array of responses already in the right shape
                    responses = this.Response(:,:,:,indices);
                end
            end
            % Cast to the right precision
            responses = this.Precision.cast( responses );
        end
        
        function dataIndices = computeDataIndices(this)
            % computeDataIndices    Compute the indices into the data from
            % start and end index
            
            dataIndices = this.StartIndexOfCurrentMiniBatch:this.EndIndexOfCurrentMiniBatch;
            
            % Convert sequential indices to ordered (possibly shuffled) indices
            dataIndices = this.OrderedIndices(dataIndices);
        end
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
    dummy = nnet.internal.cnn.util.dummify(categoricalIn);
end
end

function tensorResponses = iMatrix2Tensor( matrixResponses )
% iMatrix2Tensor   Convert a matrix of responses of size numObservations x
% numResponses to a tensor of size 1 x 1 x numResponses x numObservations
[numObservations, numResponses] = size( matrixResponses );
tensorResponses = matrixResponses';
tensorResponses = reshape(tensorResponses,[1 1 numResponses numObservations]);
end

function cats = iGetCategories(response)
cats = nnet.internal.cnn.util.categoriesFromResponse(response);
end

function imageDimensions = iGetImageDimensionsFromArray(data)
dataSize = size(data);
heightAndWidth = dataSize(1:2);
if(ismatrix(data))
    numChannels = 1;
else
    numChannels = dataSize(3);
end
imageDimensions = [heightAndWidth numChannels];
end

function responseSize = iResponseSize(response)
% iResponseSize   Return the size of the response in the first three
% dimensions.
[responseSize(1), responseSize(2), responseSize(3), ~] = size(response);
end