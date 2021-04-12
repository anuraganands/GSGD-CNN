classdef SequenceClassificationDispatcher < nnet.internal.cnn.sequence.SequenceDispatcher
    % SequenceClassificationDispatcher   Dispatch time series data for
    % classification problems one mini batch at a time from a set of time
    % series
    %
    % Input data    - cell array of time series objects, where the length
    %               of the cell array is the number of observations. 
    %               Input predictors are numObs-by-1 cell arrays which
    %               contain sequences of size: DataSize-by-S
    %               Input responses are either:
    %                   - numObs-by-1 categorical arrays
    %                   - numObs-by-1 cell arrays which contain categorical
    %                   arrays of length 1-by-S. S here must match the S of
    %                   the corresponding predictor.
    %               
    % Output data   - numeric arrays with the following dimensions:
    %               Output predictors:
    %                   - DataSize-by-MiniBatchSize-by-S
    %               Output responses are either:
    %                   - ResponseSize-by-MiniBatchSize
    %                   - ResponseSize-by-MiniBatchSize-by-S
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % DataSize (int)   Number of dimensions per time step of the input
        % data (D)
        DataSize
        
        % ResponseSize (int)   Number of classes in the response
        ResponseSize
        
        % NumObservations (int)   Number of observations in the data set
        NumObservations
        
        % SequenceLength   Strategy to determine the length of the
        % sequences used per mini-batch. Options are:
        %       - 'longest' to pad all sequences in a batch to the length
        %       of the longest sequence
        %       - 'shortest' to truncate all sequences in a batch to the
        %       length of the shortest sequence
        %       -  Positive integer - Pad sequences to the have same length
        %       as the longest sequence, then split into smaller sequences
        %       of the specified length. If splitting occurs, then the
        %       function creates extra mini-batches
        SequenceLength
        
        % DispatcherFormat (char)   Format of the response. Either:
        %       'seq2one'  : classify the entire sequence. Response is
        %       numObs-by-1 categorical array.
        %       'seq2seq'  : classify each time step. Response is
        %       numObs-by-1 cell array, containing 1-by-SequenceLength
        %       categorical arrays.        
        %       'predict'  : response is empty
        DispatcherFormat
        
        % NextStrategy (nnet.internal.cnn.sequence.NextStrategy)   Strategy
        % class which determines how mini-batches are prepared, based on:
        %       - DispatcherFormat
        %       - SequenceLength
        %       - readStrategy
        NextStrategy
        
        % Categories (categorical array) Array of categories corresponding 
        %            to training data labels. It holds also ordinality of 
        %            the labels.
        Categories = categorical();
        
        % ResponseNames (cellstr)   Array of response names corresponding
        %               to training data response names. Since we cannot
        %               get the names anywhere for an array, we will use a
        %               fixed response name.
        ResponseNames = {'Response'};
        
        % PaddingValue (scalar)
        PaddingValue
    end
    
    properties(SetAccess = private, Dependent)
        % ClassNames (cellstr) Array of class names corresponding to
        %            training data labels.
        ClassNames;
    end
    
    properties (Access = ?nnet.internal.cnn.DistributableDispatcher)
        % Data  (cell array)     A copy of the data in the workspace. This
        % is a numObservations-by-1 cell array, which for each observation
        % contains a DataSize-by-sequenceLength numeric array.
        Data
        
        % Response   A copy of the response data in the workspace. Either:
        % - numObservations-by-1 categorical vector 
        % - numObservations-by-1 cell array, which for each observation
        % contains a 1-by-sequenceLength categorical vector.
        Response
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
    
    methods
        function this = SequenceClassificationDispatcher(data, response, miniBatchSize, ...
                sequenceLength, endOfEpoch, paddingValue, precision)
            % SequenceClassificationDispatcher   Constructor for sequence
            % classification dispatcher
            %
            % data              - cell array of sequences for training. The
            %                   number of elements in the cell array is the
            %                   number of training observations. Each
            %                   sequence has dimension D x S. D is fixed
            %                   for all observations, but S may vary per
            %                   observation
            % response          - Data responses in the form of either:
            %                   numObservations x 1  categorical array
            %                   numObservations x 1 cell array. 
            %                   The cell array must contain a 1 x S
            %                   categorical vector for each observation,
            %                   with S corresponding to the S of the data
            %                   at that observation
            % miniBatchSize     - Size of a mini batch expressed in number
            %                   of examples
            % sequenceLength    - Strategy to determine the length of the
            %                   sequences used per mini-batch. Options
            %                   are:
            %                   'shortest' to truncate all sequences in a
            %                   batch to the length of the shortest
            %                   sequence (default)
            %                   'longest' to pad all sequences in a
            %                   batch to the length of the longest sequence
            %                   Integer to pad or truncate all the
            %                   sequences in a batch to a specific integer
            %                   length.
            % endOfEpoch        - Strategy to choose how to cope with a
            %                   number of observations that is not
            %                   divisible by the desired number of mini
            %                   batches. One of: 
            %                   'truncateLast' to truncate the last mini
            %                   batch
            %                   'discardLast' to discard the last mini
            %                   batch (default)
            % paddingValue      - Scalar value used to pad sequences where
            %                   necessary. The default is 0.
            % precision         - What precision to use for the dispatched
            %                   data. Values are:
            %                   'single'
            %                   'double' (default).
            
            % Assign data and response
            [dispatcherFormat, data, response] = iGetDispatcherFormat(data, response);
            [dataSize, numObservations, data] = iGetDataSize(data);
            this.Data = data;
            this.DataSize = dataSize;
            this.NumObservations = numObservations;
            [responseSize, categories, response] = iGetResponseSize(response, dispatcherFormat);
            this.Response = response;
            this.Categories = categories;
            this.ResponseSize = responseSize;
            this.DispatcherFormat = dispatcherFormat;
            
            % Assign properties
            this.SequenceLength = sequenceLength;
            this.EndOfEpoch = endOfEpoch;
            this.PaddingValue = paddingValue;
            this.Precision = precision;
            this.MiniBatchSize = miniBatchSize;
            this.OrderedIndices = 1:this.NumObservations;
            
            % Assign read strategy
            readStrategy = iGetReadStrategy( dispatcherFormat, this.ClassNames, responseSize );
            
            % Assign next strategy
            this.NextStrategy = iGetNextStrategy(dispatcherFormat, ...
                readStrategy, ...
                sequenceLength, ...
                dataSize, ...
                responseSize, ...
                paddingValue);
        end
        
        function names = get.ClassNames(this)
            names = categories(this.Categories);
            names = cellstr(names);
        end
    end
end

function [dispatcherFormat, data, response] = iGetDispatcherFormat(data, response)
if isempty(response)
    % prediction only case
    dispatcherFormat = 'predict';
    % wrap data into cell if it is numeric (single observation)
    if isnumeric( data )
        data = { data };
    end
elseif iscell(data) && iscategorical(response)
    % seq2one format
    dispatcherFormat = 'seq2one';
elseif iscell(data) && iscell(response)
    % multi-observation seq2seq format
    dispatcherFormat = 'seq2seq';
elseif isnumeric(data) && iscategorical(response)
    % seq2seq with one observation format. Wrap data/response into cells
    dispatcherFormat = 'seq2seq';
    data = { data };
    response = { response };
end
end

function [dataDimension, numObservations, data] = iGetDataSize(data)
numObservations = numel( data );
dataDimension = size( data{1}, 1 );
end

function [responseSize, cats, response] = iGetResponseSize(response, dispatcherFormat)
switch dispatcherFormat
    case 'predict'
        % Empty response => prediction only
        responseSize = [];
        cats = categorical();
    case 'seq2one'
        % Numeric response => seq2one problem
        cats = nnet.internal.cnn.util.categoriesFromResponse(response);
        responseSize = numel( categories(cats) );
    case 'seq2seq'
        % Cell array of sequences => seq2seq problem
        allResponses = [ response{:} ];
        cats = nnet.internal.cnn.util.categoriesFromResponse(allResponses);        
        classNames = categories( cats );
        responseSize = numel( classNames );
        for ii = 1:numel( response )
            response{ii} = setcats( response{ii}, classNames );
        end
end
end

function strategy = iGetNextStrategy(dispatcherFormat, readStrategy, sequenceLength, dataSize, responseSize, paddingValue)
switch dispatcherFormat
    case 'seq2seq'
        switch sequenceLength
            case 'longest'
                strategy = nnet.internal.cnn.sequence.Seq2SeqLongestStrategy(readStrategy, dataSize, responseSize, paddingValue);
            case 'shortest'
                strategy = nnet.internal.cnn.sequence.Seq2SeqShortestStrategy(readStrategy, dataSize, responseSize, paddingValue);
            otherwise
                strategy = nnet.internal.cnn.sequence.Seq2SeqFixedStrategy(readStrategy, dataSize, responseSize, paddingValue);
        end
    case {'seq2one', 'predict'}
        switch sequenceLength
            case 'longest'
                strategy = nnet.internal.cnn.sequence.Seq2OneLongestStrategy(readStrategy, dataSize, responseSize, paddingValue);
            case 'shortest'
                strategy = nnet.internal.cnn.sequence.Seq2OneShortestStrategy(readStrategy, dataSize, responseSize, paddingValue);
            otherwise
                strategy = nnet.internal.cnn.sequence.Seq2OneFixedStrategy(readStrategy, dataSize, responseSize, paddingValue);
        end
end
end

function strategy = iGetReadStrategy( dispatcherFormat, classNames, responseSize )
strategy.readDataFcn = iReadData();
strategy.readResponseFcn = iReadResponse( dispatcherFormat, classNames, responseSize );
end

function fcn = iReadData()
fcn = @(data, indices)data( indices );
end

function fcn = iReadResponse( dispatcherFormat, classNames, responseSize )
switch dispatcherFormat
    case 'predict'
        fcn = @(response, indices)[];
    case 'seq2one'
        fcn = @(response, indices)iDummify( response( indices ), classNames, responseSize );
    case 'seq2seq'
        fcn = @(response, index)iDummify( response{ index }, classNames, responseSize );
end
end

function dummy = iDummify(categoricalIn, classNames, responseSize)
% dummify   Dummify a categorical vector of size numObservations x 1 to
% return a matrix of size numClasses x numObservations
categoricalIn = reordercats( categoricalIn, classNames );
numObservations = numel(categoricalIn);
dummifiedSize = [responseSize, numObservations];
dummy = zeros(dummifiedSize);
categoricalIn = reshape( categoricalIn, 1, numel( categoricalIn ) );
idx = sub2ind(dummifiedSize, single(categoricalIn), 1:numObservations);
idx(isnan(idx)) = [];
dummy(idx) = 1;
end