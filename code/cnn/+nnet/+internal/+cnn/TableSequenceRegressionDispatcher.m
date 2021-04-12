classdef TableSequenceRegressionDispatcher < nnet.internal.cnn.sequence.SequenceDispatcher
    % TableSequenceRegressionDispatcher   Dispatch out-of-memory time
    % series data for regression problems one mini batch at a time from
    % a set of time series
    %
    % Input data    - Table of input predictors and responses. Predictors
    %               are specified with MAT file path locations. Responses
    %               can be numObs-by-responseSize numeric array, or MAT
    %               file path locations to a sequence response. A MAT file
    %               used to specify predictors or responses must contain a
    %               numeric array of size dataSize-by-S as its first
    %               quantity.
    %               
    % Output data   - numeric arrays with the following dimensions:
    %               Output predictors:
    %                   - DataSize-by-MiniBatchSize-by-S
    %               Output responses are either:
    %                   - ResponseSize-by-MiniBatchSize
    %                   - ResponseSize-by-MiniBatchSize-by-S
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % DataSize (int)   Number of dimensions per time step of the input
        %                   data (D)
        DataSize
        
        % ResponseSize (int)   Number of elements in the response
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
        %       'seq2one'  : regress the entire sequence. Response is
        %       numObs-by-responseSize numeric array.
        %       'seq2seq'  : regress each time step. Response is
        %       numObs-by-1 cell array, containing
        %       responseSize-by-SequenceLength numeric arrays.
        %       'predict'      : response is empty
        DispatcherFormat
        
        % NextStrategy (nnet.internal.cnn.sequence.NextStrategy)   Strategy
        % class which determines how mini-batches are prepared, based on:
        % - the dispatcher format
        % - the sequenceLength argument
        % - the read strategy
        NextStrategy
        
        % ClassNames (cellstr)   Array of class names corresponding to
        %            training data labels.
        ClassNames = {};
        
        % ResponseNames (cellstr)   Array of response names corresponding
        %               to training data response names. Since we cannot
        %               get the names anywhere for an array, we will use a
        %               fixed response name.
        ResponseNames = {'Response'};
        
        % PaddingValue (scalar)
        PaddingValue
        
        % DataTable (table)   A table whose first column holds file paths
        % to predictors saved as MAT files, and whose second column
        % contains responses
        DataTable
    end
    
    properties (Access = ?nnet.internal.cnn.DistributableDispatcher)
        % Data  (filepath datastore)   A filepath datastore of the
        % predictor sequences.
        Data
        
        % Response   A copy of the response data. Either:
        % - (seq2one) numObservations-by-responseSize numeric array
        % - (seq2seq) filepath datastore of the response sequences
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
        function this = TableSequenceRegressionDispatcher(dataTable, miniBatchSize, ...
                sequenceLength, endOfEpoch, paddingValue, precision, dataSize, networkResponseSize)
            % TableSequenceRegressionDispatcher   Constructor for sequence
            % regression dispatcher with file-path table input
            %
            % dataTable         - table of predictors and responses for
            %                   training. The first column of the table
            %                   must be file-path locations of the
            %                   predictors. The second column must contain
            %                   the responses. The responses can be a
            %                   numObservations x numResponses numeric
            %                   array, or file-path locations to sequences.
            % miniBatchSize     - Size of a mini batch expressed in number
            %                   of examples.
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
            % dataSize          - Positive integer stating the data
            %                   dimension of the predictors.
            % networkResponseSize      - Positive integer stating the data
            %                   dimension of the responses, inferred from
            %                   the network.
            
            % Assign data and response
            [numObservations, predictorFiles, dispatcherFormat] = iReadDataTable(dataTable);
            this.DataTable = dataTable;
            this.Data = fileDatastore( fullfile(predictorFiles), ...
                'ReadFcn', @load, 'FileExtensions', '.mat' );
            this.DataSize = dataSize;
            this.NumObservations = numObservations;
            this.DispatcherFormat = dispatcherFormat;
            this.Response = iCreateResponse(dispatcherFormat, dataTable);
            this.ResponseSize = iGetResponseSize(dispatcherFormat, this.Response, networkResponseSize);
            
            % Assign properties
            this.SequenceLength = sequenceLength;
            this.EndOfEpoch = endOfEpoch;
            this.PaddingValue = paddingValue;
            this.Precision = precision;
            this.MiniBatchSize = miniBatchSize;
            this.OrderedIndices = 1:this.NumObservations;
            
            % Assign read strategy
            readStrategy = iGetReadStrategy( dispatcherFormat );
            
            % Assign next strategy
            this.NextStrategy = iGetNextStrategy(dispatcherFormat, ...
                readStrategy, ...
                sequenceLength, ...
                dataSize, ...
                networkResponseSize, ...
                paddingValue);
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
    otherwise
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

function strategy = iGetReadStrategy( dispatcherFormat )
strategy.readDataFcn = iReadData();
strategy.readResponseFcn = iReadResponse( dispatcherFormat );
end

function X = iReadDatastore(datastore, indices)
% Create datastore partition via a copy and index. This is
% faster than constructing a new datastore with the new
% files.
subds = copy(datastore);
subds.Files = datastore.Files(indices);
X = subds.readall();
% Convert to array
X = cellfun(@iReadDataFromStruct, X, 'UniformOutput', false);
end

function fcn = iReadData()
fcn = @(data, indices)iReadDatastore(data, indices);
end

function Y = iReadSeq2SeqResponse( response, index )
Ycell = iReadDatastore( response, index );
Y = Ycell{:};
end

function fun = iReadResponse( dispatcherFormat )
switch dispatcherFormat
    case 'predict'
        fun = @(response, indices)[];
    case 'seq2one'
        fun = @(response, indices)response( indices, : )';
    case 'seq2seq'
        fun = @(response, index)iReadSeq2SeqResponse( response, index );
end
end

function [numObservations, dataFilePaths, responseFormat] = iReadDataTable(dataTable)
if istable( dataTable )
    numObservations = height( dataTable );
    % File paths are in the first column by assumption
    dataFilePaths = dataTable{:, 1};
    if width( dataTable ) == 1
        responseFormat = 'predict';
    else
        % Responses are in the second column and onwards by assumption
        response = dataTable{:, 2:end};
        responseFormat = iGetDispatcherFormat(response);
    end
else
    % Error - input is not a table
end
end

function dispatcherFormat = iGetDispatcherFormat(response)
if isempty( response )
    dispatcherFormat = 'predict';
elseif isreal( response )
    dispatcherFormat = 'seq2one';
elseif iscell( response )
    % Assume cell array of file paths
    dispatcherFormat = 'seq2seq';
end
end

function data = iReadDataFromStruct(S)
% Read data from first field in struct S
fn = fieldnames( S );
data = S.(fn{1});
% Make sure data is not empty
if isempty( data )
    error(message('nnet_cnn:internal:cnn:TableSequenceRegressionDispatcher:XIsNotValidSequenceInput'));
end
end

function response = iCreateResponse(dispatcherFormat, dataTable)
% iCreateResponse   Create appropriate response based on dispatcher format
switch dispatcherFormat
    case 'predict'
        response = [];
    case 'seq2one'
        response = dataTable{:, 2:end};
    case 'seq2seq'
        responseFiles = dataTable{:, 2};
        response = fileDatastore( fullfile(responseFiles), ...
            'ReadFcn', @load, 'FileExtensions', '.mat' );
end
end

function responseSize = iGetResponseSize(dispatcherFormat, response, networkResponseSize)
% iGetResponseSize   Get the size of the response from the data, if the
% response is in-memory
switch dispatcherFormat
    case 'predict'
        responseSize = networkResponseSize;
    case 'seq2one'
        responseSize = size( response, 2 );
    case 'seq2seq'
        responseSize = networkResponseSize;
end
end