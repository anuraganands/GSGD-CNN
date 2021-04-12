classdef DataDispatcherFactory
    % DataDispatcherFactory   Factory for making data dispatchers
    %
    %   dataDispatcher = DataDispatcherFactoryInstance.createDataDispatcher(data, options)
    %   data: the data to be dispatched.
    %       According to their type the appropriate dispatcher will be used.
    %       Supported types: 4-D double, imagedatastore, table
    %   options: input arguments for the data dispatcher (e.g. response vector,
    %   mini batch size)
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    methods (Static)
        function dispatcher = createDataDispatcher( inputs, response, ...
                miniBatchSize, endOfEpoch, precision, executionSettings, ...
                shuffleOption, sequenceLength, paddingValue, layers )
            % createDataDispatcher   Create data dispatcher
            %
            % Syntax:
            %     createDataDispatcher( inputs, response, ... 
            % miniBatchSize, endOfEpoch, precision, executionSettings, ...
            % shuffleOption, sequenceLength, paddingValue, layers )
            
            % Allow executionSettings to be unspecified
            if nargin < 6
                executionSettings = struct( 'useParallel', false );
            end
            % Dispatch in background is requested by input, unless
            % overridden by executionSettings
            if isfield(executionSettings, 'backgroundPrefetch')
                backgroundPrefetch = executionSettings.backgroundPrefetch;
            else
                backgroundPrefetch = iDataRequestsBackgroundPrefetch(inputs);
            end
            % Allow shuffle setting to be unspecified
            if nargin < 7
                shuffleOption = 'once';
            end
            % Allow sequenceLength to be unspecified
            if nargin < 8
                sequenceLength = 'longest';
            end
            % Allow paddingValue to be unspecified
            if nargin < 9
                paddingValue = 0;
            end
            % Allow layers to be unspecified
            if nargin < 10
                layers = nnet.cnn.layer.Layer.empty();
            end
            
            if isa(inputs, 'nnet.internal.cnn.DataDispatcher')
                dispatcher = inputs;
                
                % Setup the dispatcher to factory specifications.
                dispatcher.EndOfEpoch    = endOfEpoch;
                dispatcher.Precision     = precision;
                dispatcher.MiniBatchSize = miniBatchSize; 
            else
                isRNN = iIsRNN( layers );
                isAClassificationNetwork = iIsAClassificationNetwork( layers );
                if iIsRealNumeric4DHostArray(inputs) && ~isRNN
                    datasource = iCreate4dArrayMiniBatchDatastore( inputs, response, miniBatchSize );
                elseif isa(inputs, 'matlab.io.datastore.ImageDatastore')
                    datasource  = iCreateImageDatastoreMiniBatchDatastore( inputs, miniBatchSize );
                elseif iIsAMiniBatchDatastore(inputs)
                    nnet.internal.cnn.validateMiniBatchDatastore(inputs);
                    datasource = inputs;
                elseif istable(inputs) && ~isRNN
                    if iIsAnInMemoryTable(inputs)
                        datasource = iCreateInMemoryTableMiniBatchDatastore( inputs, miniBatchSize );
                    else
                        datasource  = iCreateFilePathTableMiniBatchDatastore( inputs, miniBatchSize );
                    end
                elseif isRNN && iIsSequenceInMemoryInput(inputs)
                    if isAClassificationNetwork
                        dispatcher = iCreateSequenceClassificationDispatcher( inputs, response, ...
                            miniBatchSize, sequenceLength, endOfEpoch, paddingValue, precision );
                    else
                        dispatcher = iCreateSequenceRegressionDispatcher( inputs, response, ...
                            miniBatchSize, sequenceLength, endOfEpoch, paddingValue, precision );
                    end
                    datasource = [];
                elseif isRNN && istable(inputs)
                    if isAClassificationNetwork
                        dispatcher = iCreateTableSequenceClassificationDispatcher( inputs, miniBatchSize, ...
                            sequenceLength, endOfEpoch, paddingValue, precision, layers );
                    else
                        dispatcher = iCreateTableSequenceRegressionDispatcher( inputs, miniBatchSize, ...
                            sequenceLength, endOfEpoch, paddingValue, precision, layers );
                    end
                    datasource = [];
                else
                    error( message( 'nnet_cnn:internal:cnn:DataDispatcherFactory:InvalidData' ) );
                end
                
                if executionSettings.useParallel && iIsAMiniBatchDatastore(datasource) &&...
                        ~isa(datasource,'matlab.io.datastore.PartitionableByIndex')
                    
                    error(message('nnet_cnn:internal:cnn:DataDispatcherFactory:NonDistributableMiniBatchDatastore'));
                end
                                    
                if iIsAMiniBatchDatastore(datasource)
                    dispatcher = nnet.internal.cnn.MiniBatchDatastoreDispatcher( datasource, miniBatchSize, endOfEpoch, precision );
                end
                
            end
            
            % If the dispatcher type doesn't support background, ensure it
            % is disabled. Similarly, if it does support it, make sure it
            % is enabled or disabled as appropriate.
            if ~isa(dispatcher, 'nnet.internal.cnn.BackgroundCapableDispatcher')
                backgroundPrefetch = false;
            else
                if backgroundPrefetch
                    dispatcher.setRunInBackground(true);
                else
                    dispatcher.setRunInBackground(false);
                end
            end
                       
            % Create BackgroundDispatcher wrapper for serial dispatch
            if backgroundPrefetch && ~executionSettings.useParallel
                dispatcher = nnet.internal.cnn.BackgroundDispatcher( dispatcher );
            end
                                                                             
            % Distribute for Parallel. Status of background prefetch is
            % recorded in the dispatcher's RunInBackground property.
            if executionSettings.useParallel
                retainDataOrder = isequal(shuffleOption, 'never');
                dispatcher = nnet.internal.cnn.DistributedDispatcher( dispatcher, executionSettings.workerLoad, retainDataOrder );
            end
        end
    end
end

function ds = iCreate4dArrayMiniBatchDatastore( inputs, response, miniBatchSize )
ds = nnet.internal.cnn.FourDArrayMiniBatchDatastore(inputs, response, miniBatchSize);
end

function ds = iCreateImageDatastoreMiniBatchDatastore( inputs, miniBatchSize )
ds = nnet.internal.cnn.ImageDatastoreMiniBatchDatastore( inputs, miniBatchSize );
end

function ds = iCreateFilePathTableMiniBatchDatastore( inputs, miniBatchSize )
ds = nnet.internal.cnn.FilePathTableMiniBatchDatastore( inputs, miniBatchSize );
end

function ds = iCreateInMemoryTableMiniBatchDatastore( inputs, miniBatchSize )
ds = nnet.internal.cnn.InMemoryTableMiniBatchDatastore( inputs, miniBatchSize );
end

function ds = iCreateSequenceClassificationDispatcher( inputs, response, ...
    miniBatchSize, sequenceLength, endOfEpoch, paddingValue, precision )
ds = nnet.internal.cnn.SequenceClassificationDispatcher( inputs, response, ...
    miniBatchSize, sequenceLength, endOfEpoch, paddingValue, precision );              
end

function ds = iCreateSequenceRegressionDispatcher( inputs, response, ...
    miniBatchSize, sequenceLength, endOfEpoch, paddingValue, precision )
ds = nnet.internal.cnn.SequenceRegressionDispatcher( inputs, response, ...
    miniBatchSize, sequenceLength, endOfEpoch, paddingValue, precision );              
end

function ds = iCreateTableSequenceClassificationDispatcher( inputs, miniBatchSize, ...
    sequenceLength, endOfEpoch, paddingValue, precision, layers )
inputSize = layers(1).InputSize;
outputSize = layers(end).OutputSize;
ds = nnet.internal.cnn.TableSequenceClassificationDispatcher( inputs, miniBatchSize, ...
    sequenceLength, endOfEpoch, paddingValue, precision, inputSize, outputSize );            
end

function ds = iCreateTableSequenceRegressionDispatcher( inputs, miniBatchSize, ...
    sequenceLength, endOfEpoch, paddingValue, precision, layers )
inputSize = layers(1).InputSize;
outputSize = iGetRegressionLayersOutputSize( layers );
ds = nnet.internal.cnn.TableSequenceRegressionDispatcher( inputs, miniBatchSize, ...
    sequenceLength, endOfEpoch, paddingValue, precision, inputSize, outputSize );       
end

function tf = iIsRealNumeric4DHostArray( x )
tf = iIsRealNumericData( x ) && iIsValidImageArray( x ) && ~iIsGPUArray( x );
end

function tf = iIsRealNumericData(x)
tf = isreal(x) && isnumeric(x) && ~issparse(x);
end

function tf = iIsValidImageArray(x)
% iIsValidImageArray   Return true if x is an array of
% one or multiple (colour or grayscale) images
tf = ( iIsGrayscale( x ) || iIsColour( x ) || iIsMultiChannel( x ) ) && ...
    iIs4DArray( x ) ;
end

function tf = iIsGrayscale(x)
tf = size(x,3)==1;
end

function tf = iIsMultiChannel(x)
tf = size(x,3) > 1;
end

function tf = iIsColour(x)
tf = size(x,3)==3;
end

function tf = iIs4DArray(x)
sz = size( x );
tf = numel( sz ) <= 4;
end

function tf = iIsGPUArray( x )
tf = isa(x, 'gpuArray');
end

function tf = iIsAnInMemoryTable( x )
firstCell = x{1,1};
tf = isnumeric( firstCell{:} );
end

function tf = iIsSequenceInMemoryInput( x )
tf = (iscell(x) && isvector(x) && all(cellfun(@isnumeric, x))) || ...
    (isnumeric(x) && ismatrix(x));
end

function tf = iIsRNN( layers )
internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers( layers );
tf = nnet.internal.cnn.util.isRNN( internalLayers );
end

function tf = iIsAClassificationNetwork( layers )
tf = false;
if ~isempty( layers )
    tf = ( isa( layers(end), 'nnet.cnn.layer.ClassificationOutputLayer' ) || ...
        isa( layers(end), 'nnet.layer.ClassificationLayer' ) );
end
end

function outputSize = iGetRegressionLayersOutputSize( layers )
% Note -- This code is added to support LSTM regression SeriesNetworks.
% Modification will be required when DAG support is added.
internalLayers = nnet.cnn.layer.Layer.getInternalLayers( layers ); 
outputSize = internalLayers{1}.InputSize;
for i = 2:numel(layers) 
    outputSize = internalLayers{i}.forwardPropagateSize(outputSize); 
end 
end

function tf = iDataRequestsBackgroundPrefetch(X)
tf = (isa(X, 'matlab.io.datastore.BackgroundDispatchable') && X.DispatchInBackground) || ...
    (isa(X, 'nnet.internal.cnn.BackgroundCapableDispatcher') && X.RunInBackground);
end

function tf = iIsAMiniBatchDatastore(X)
    tf = isa(X,'matlab.io.Datastore') && isa(X,'matlab.io.datastore.MiniBatchable');
end