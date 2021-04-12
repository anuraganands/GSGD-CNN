classdef (Sealed) NetworkDataValidator
    % NetworkDataValidator   Class that holds various validation functions
    % for data with trainNetwork, SeriesNetwork and DAGNetwork.
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties (Access = private)
        % ErrorStrategy (nnet.internal.cnn.util.ErrorStrategy)   Error strategy
        ErrorStrategy
    end
    
    methods
        function this = NetworkDataValidator( errorStrategy )
            this.ErrorStrategy = errorStrategy;
        end
        
        function validateDataForProblem( this, X, Y, layers )
            % validateDataForProblem   Assert that the input data X and
            % response Y are valid for the class of problem considered
            internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers( layers );
            isClassificationNetwork = iIsInternalClassificationNetwork(internalLayers);
            isRNN = iIsRNN( internalLayers );
            if iIsAnImageDatastore( X ) && ~isRNN
                this.assertValidInputImageDatastore( X );
                this.assertClassificationForImageDatastore(isClassificationNetwork);
            elseif istable( X )
                this.assertValidTable( X, isClassificationNetwork, isRNN, internalLayers );
            elseif isnumeric( X ) && ~isRNN
                this.assertValidImageArray( X );
                this.assertValidResponseForTheNetwork( Y, isClassificationNetwork );
                this.assertXAndYHaveSameNumberOfObservations( X, Y );
            elseif iIsADataDispatcher( X )
                % X is a custom dispatcher - the custom dispatcher api is
                % for internal use only
            elseif isRNN && isClassificationNetwork
                this.assertNotDatastoreOrSource( X );
                this.assertValidSequenceInput( X );
                this.assertValidSequenceResponse( Y, true );
                this.assertOutputModeCorrespondsToDataForClassification( X, Y, internalLayers );
                this.assertResponsesHaveValidSequenceLength( X, Y );
                this.assertSequencesHaveSameNumberOfObservations( X, Y );
            elseif isRNN && ~isClassificationNetwork
                this.assertNotDatastoreOrSource( X );
                this.assertValidSequenceInput( X );
                this.assertValidSequenceResponse( Y, false );
                this.assertOutputModeCorrespondsToDataForRegression( X, Y, internalLayers );
                this.assertResponsesHaveValidSequenceLength( X, Y );
                this.assertSequencesHaveSameNumberOfObservations( X, Y );    
            elseif iIsAMiniBatchableDatastore( X )
                
            else
                this.ErrorStrategy.throwXIsNotValidType();
            end
        end
        
        function validateDataForPredict( this, X, dispatcher, layers, isRNN )
            % validateDataForPredict   Assert that input data X is valid
            % for the SeriesNetwork predict() method
            
            % NB -- branches for CNNs should be added to this logic
            if iIsADataDispatcher( X )
                % X is a custom dispatcher - the custom dispatcher api is
                % for internal use only
            elseif isRNN
                this.assertNotDatastoreOrSource( X );
                this.assertValidSequenceInput( X );
                internalInputLayer = nnet.cnn.layer.Layer.getInternalLayers( layers(1) );
                this.assertCorrectDataSizeForInputLayer(dispatcher, internalInputLayer{:});
            else
                this.ErrorStrategy.throwXIsNotValidType();
            end
        end
        
        function validateDataSizeForNetwork( this, dispatcher, internalLayers )
            % validateDataSizeForNetwork   Assert that the data to be
            % dispatched by dispatcher is of size that fits the network
            % represented by internalLayers
            this.assertCorrectDataSizeForInputLayer(dispatcher, internalLayers{1})
            this.assertCorrectResponseSizeForOutputLayer(dispatcher, internalLayers)
        end
        
        function validateDataSizeForDAGNetwork( this, dispatcher, lgraph )
            % validateDataSizeForDAGNetwork   Assert that the data to be
            % dispatched by dispatcher is of size that fits the network
            % represented by lgraph
            internalLayers = iGetInternalLayers(lgraph.Layers);
            this.assertCorrectDataSizeForInputLayer(dispatcher, internalLayers{1})
            this.assertCorrectResponseSizeForDAGOutputLayer(dispatcher, lgraph, internalLayers)
        end
    end
    
    methods (Access = private)
        %% Assert helpers
        function assertLabelsAreDefined(this, labels)
            if iscategorical( labels )
                if(any(isundefined(labels)))
                    this.ErrorStrategy.throwUndefinedLabels()
                end
            else 
                % Assume filepath labels. Load the first response and check
                % for undefined labels. Loading all responses may be
                % time-consuming, so we load only the first
                D = load( labels{1} );
                response = iReadDataFromStruct( D );
                if(any(isundefined(response)))
                    this.ErrorStrategy.throwUndefinedLabels()
                end
            end
        end
        
        function assertValidImageArray(this, x)
            if isa(x, 'gpuArray') || ~isnumeric(x) || ~isreal(x) || ~iIsValidImageArray(x)
                this.ErrorStrategy.throwXIsNotValidImageArray()
            end
        end
        
        function assertValidRegressionResponse(this, x)
            if isa(x, 'gpuArray') || ~isnumeric(x) || ~isreal(x) || ~iIsValidResponseArray(x)
                this.ErrorStrategy.throwYIsNotValidResponseArray()
            end
        end
        
        function assertValidResponseForTheNetwork(this, x, isAClassificationNetwork)
            % assertValidResponseForTheNetwork   Assert if x is a valid response for
            % the type of network in use.
            if isAClassificationNetwork
                this.assertCategoricalResponseVector(x);
                this.assertLabelsAreDefined(x);
            else
                this.assertValidRegressionResponse(x);
            end
        end
        
        function assertCategoricalResponseVector(this, x)
            if ~(iscategorical(x) && isvector(x))
                this.ErrorStrategy.throwYIsNotCategoricalResponseVector()
            end
        end
        
        function assertXAndYHaveSameNumberOfObservations(this, x, y)
            if size(x,4)~=iArrayResponseNumObservations(y)
                this.ErrorStrategy.throwXAndYHaveDifferentObservations()
            end
        end
        
        function assertClassificationForImageDatastore(this, isAClassificationNetwork)
            if ~isAClassificationNetwork
                this.ErrorStrategy.throwImageDatastoreWithRegression()
            end
        end
        
        function assertValidTable(this, tbl, isAClassificationNetwork, isRNN, internalLayers)
            % assertValidTable   Assert that tbl is a valid table according to the
            % type of network defined by layers (classification or regression).
            
            if isRNN
                this.assertValidRNNPredictors(tbl);
                networkSeq2Seq = iReturnsSequence( internalLayers, isRNN );
                dataSeq2One = iHasValidSeq2OneResponseColumn(tbl);
                dataSeq2Seq = iHasValidSeq2SeqResponseColumn(tbl);
                if dataSeq2One && networkSeq2Seq
                    this.ErrorStrategy.throwOutputModeSequenceTableDataMismatch();
                elseif dataSeq2Seq && ~networkSeq2Seq
                    this.ErrorStrategy.throwOutputModeLastTableDataMismatch();
                elseif networkSeq2Seq && ~dataSeq2Seq
                    this.assertValidSeq2SeqTableResponses(tbl);
                elseif ~networkSeq2Seq && ~dataSeq2One
                    this.assertValidSeq2OneTableResponses(tbl);
                end
            else
                if isAClassificationNetwork
                    this.assertValidClassificationTable(tbl);
                else
                    this.assertValidRegressionTable(tbl);
                end
            end
        end
        
        function assertValidClassificationTable(this, tbl)
            % assertValidClassificationTable   Assert that tbl is a valid
            % classification table. To be valid, it needs to have image paths or images
            % in the first column. Responses will be held in the second column as
            % categorical labels.
            isValidFirstColumn = iHasValidPredictorColumn(tbl);
            hasValidResponses = iHasValidClassificationResponses(tbl);
            isValidClassificationTable = isValidFirstColumn && hasValidResponses;
            if ~isValidClassificationTable
                this.ErrorStrategy.throwInvalidClassificationTable()
            end
            % Assert that all labels are defined for the classification problem
            this.assertLabelsAreDefined(tbl{:,2});
        end
        
        function assertValidRegressionTable(this, tbl)
            % assertValidRegressionTable   Assert that tbl is a valid regression
            % table. To be valid, it needs to have image paths or images in the first
            % column. Responses will be held in the second column as either vectors or
            % cell arrays containing 3-D arrays. Alternatively, responses will be held
            % in multiple columns as scalars.
            if ~iHasValidPredictorColumn(tbl)
                this.ErrorStrategy.throwInvalidRegressionTablePredictors()
            end
            if  ~iHasValidRegressionResponses(tbl)
                this.ErrorStrategy.throwInvalidRegressionTableResponses()
            end
        end
        
        function assertValidRNNPredictors(this, tbl)
            % assertValidRNNPredictors   Assert that tbl contains only
            % valid RNN predictors -- file paths to sequences.
            if ~iFirstColumnContainsOnlyPaths(tbl)
                this.ErrorStrategy.throwInvalidRNNTablePredictors()
            end
        end
        
        function assertValidSeq2OneTableResponses(this, tbl)
            % assertValidSeq2OneTableResponse   Assert that tbl has valid
            % seq2one responses, with in-memory data in the second column
            if  ~iHasValidSeq2OneResponseColumn(tbl)
                this.ErrorStrategy.throwInvalidSeq2OneTableResponse()
            end
        end
        
        function assertValidSeq2SeqTableResponses(this, tbl)
            % assertValidSeq2SeqTableResponse   Assert that tbl has valid
            % seq2seq responses, with file paths in the second column.
            if  ~iHasValidSeq2SeqResponseColumn(tbl)
                this.ErrorStrategy.throwInvalidSeq2SeqTableResponse()
            end
        end

        function assertValidInputImageDatastore(this, X)
            % assertValidInputImageDatastore   Assert that X is a valid image
            % datastore to be used in trainNetwork
            this.assertValidImageDatastore( X );
            this.assertDatastoreHasLabels( X );
            this.assertDatastoreLabelsAreCategorical( X );
            this.assertLabelsAreDefined( X.Labels );
        end
        
        function assertValidImageDatastore(this, imds)
            if ~iIsAnImageDatastore(imds)
                this.ErrorStrategy.throwNotAnImageDatastore()
            end
        end
        
        function assertDatastoreHasLabels(this, imds)
            if isempty(imds.Labels)
                this.ErrorStrategy.throwImageDatastoreHasNoLabels()
            end
        end
        
        function assertDatastoreLabelsAreCategorical(this, imds)
            if ~iscategorical(imds.Labels)
                this.ErrorStrategy.throwImageDatastoreMustHaveCategoricalLabels()
            end
        end
        
        function assertCorrectDataSizeForInputLayer(this, dispatcher, internalInputLayer)
            if isa( internalInputLayer, 'nnet.internal.cnn.layer.ImageInput' )
                if(~internalInputLayer.isValidTrainingImageSize(dispatcher.ImageSize))
                    this.ErrorStrategy.throwImagesInvalidSize( ...
                        i3DSizeToString(dispatcher.ImageSize), ...
                        i3DSizeToString(internalInputLayer.InputSize) );
                end
            elseif isa( internalInputLayer, 'nnet.internal.cnn.layer.SequenceInput' )
                if(~internalInputLayer.isValidInputSize(dispatcher.DataSize))
                    this.ErrorStrategy.throwSequencesInvalidSize( ...
                        int2str(dispatcher.DataSize), ...
                        int2str(internalInputLayer.InputSize) );
                end
            end
        end
        
        function assertCorrectResponseSizeForOutputLayer(this, dispatcher, internalLayers)
            networkOutputSize = iNetworkOutputSize(internalLayers);
            dataResponseSize = dispatcher.ResponseSize;
            
            if ~isequal( networkOutputSize, dataResponseSize )
                if iIsInternalClassificationNetwork(internalLayers)
                    % For a classification network, we output a "classes
                    % mismatch" error
                    networkClasses = networkOutputSize(end);
                    expectedClasses = dataResponseSize(end);
                    this.ErrorStrategy.throwOutputSizeNumClassesMismatch( ...
                        mat2str( networkClasses ), mat2str( expectedClasses ) );
                elseif iIsScalarResponseOrSeq2OneRNN( networkOutputSize, dataResponseSize, internalLayers )
                    % If both responses are scalar, or the network is a
                    % seq2one RNN, output a different error message
                    % regarding number of responses
                    networkResponses = networkOutputSize(end);
                    dataResponses = dataResponseSize(end);
                    this.ErrorStrategy.throwOutputSizeNumResponsesMismatch( ...
                        mat2str( networkResponses ), mat2str( dataResponses ) );
                else
                    % Otherwise, we throw a response size mismatch error.
                    % For example, if the output is a tensor for
                    % super-resolution, or a seq2seq regression response
                    this.ErrorStrategy.throwOutputSizeResponseSizeMismatch( ...
                        mat2str( networkOutputSize ), mat2str( dataResponseSize ) );
                end
            end
        end
        
        function assertCorrectResponseSizeForDAGOutputLayer(this, dispatcher, lgraph, internalLayers)
            networkOutputSize = iDAGNetworkOutputSize(lgraph);
            dataResponseSize = dispatcher.ResponseSize;
            
            if ~isequal( networkOutputSize, dataResponseSize )
                if iIsInternalClassificationNetwork(internalLayers)
                    networkClasses = networkOutputSize(3);
                    expectedClasses = dataResponseSize(3);
                    this.ErrorStrategy.throwOutputSizeNumClassesMismatch( ...
                        mat2str( networkClasses ), mat2str( expectedClasses ) );
                else
                    if iIsScalarResponseSize( networkOutputSize ) && iIsScalarResponseSize( dataResponseSize )
                        % If both responses are scalar, output a different error
                        % message regarding number of responses
                        networkResponses = networkOutputSize(3);
                        dataResponses = dataResponseSize(3);
                        this.ErrorStrategy.throwOutputSizeNumResponsesMismatch( ...
                            mat2str( networkResponses ), mat2str( dataResponses ) );
                    else
                        this.ErrorStrategy.throwOutputSizeResponseSizeMismatch( ...
                            mat2str( networkOutputSize ), mat2str( dataResponseSize ) );
                    end
                end
            end
        end
        
        function assertNotDatastoreOrSource(this, x)
            if iIsAnImageDatastore(x) || iIsAMiniBatchableDatastore(x)
                this.ErrorStrategy.throwIncompatibleInputForRNN()
            end
        end
        
        function assertValidSequenceInput(this, x)
            if ~iIsValidSequenceInput(x)
                this.ErrorStrategy.throwXIsNotValidSequenceInput()
            end
            if ~iSequencesHaveConsistentDataDimension(x)
                this.ErrorStrategy.throwXIsNotValidSequenceInput()
            end
        end
        
        function assertValidSequenceResponse(this, x, isAClassificationNetwork)
            if isAClassificationNetwork
                if ~iIsValidSequenceCategoricalResponse(x)
                    this.ErrorStrategy.throwYIsNotValidSequenceCategorical()
                end
                if ~iHasDefinedSequenceLabels(x)
                    this.ErrorStrategy.throwUndefinedLabels()
                end
            else
                if ~iIsValidSequenceRegressionResponse(x)
                    this.ErrorStrategy.throwYIsNotValidSequenceResponse()
                end
                if ~iSequencesHaveConsistentDataDimension(x)
                    this.ErrorStrategy.throwYIsNotValidSequenceResponse()
                end
            end
        end
        
        function assertSequencesHaveSameNumberOfObservations(this, x, y)
            if ~iSequencesHaveSameNumberOfObservations(x, y)
                this.ErrorStrategy.throwXAndYHaveDifferentObservations()
            end
        end
        
        function assertResponsesHaveValidSequenceLength(this, x, y)
            if ~iSequencesHaveConsistentSequenceLength(x, y)
                this.ErrorStrategy.throwInvalidResponseSequenceLength()
            end
        end
        
        function assertOutputModeCorrespondsToDataForClassification(this, x, y, internalLayers)
            returnsSequence = iReturnsSequence( internalLayers, true );
            if ~iOutputModeMatchesDataForClassification(x, y, returnsSequence)
                if returnsSequence
                    this.ErrorStrategy.throwOutputModeSequenceDataMismatchClassification()
                else
                    this.ErrorStrategy.throwOutputModeLastDataMismatchClassification()
                end
            end
        end
        
        function assertOutputModeCorrespondsToDataForRegression(this, x, y, internalLayers)
            returnsSequence = iReturnsSequence( internalLayers, true );
            if ~iOutputModeMatchesDataForRegression(x, y, returnsSequence)
                if returnsSequence
                    this.ErrorStrategy.throwOutputModeSequenceDataMismatchRegression()
                else
                    this.ErrorStrategy.throwOutputModeLastDataMismatchRegression()
                end
            end
        end
        
    end
end

%% ISA/HASA helpers
function tf = iIsInternalClassificationNetwork(internalLayers)
tf = iIsInternalClassificationLayer( internalLayers{end} );
end

function tf = iIsInternalClassificationLayer(internalLayer)
tf = isa(internalLayer, 'nnet.internal.cnn.layer.ClassificationLayer');
end

function tf = iIsRNN(internalLayers)
tf = nnet.internal.cnn.util.isRNN( internalLayers );
end

function tf = iReturnsSequence(internalLayers, isRNN)
tf = nnet.internal.cnn.util.returnsSequence(internalLayers, isRNN);
end

function tf = iIsAnImageDatastore(x)
tf = isa(x, 'matlab.io.datastore.ImageDatastore');
end

function tf = iIsADataDispatcher(X)
tf = isa(X, 'nnet.internal.cnn.DataDispatcher');
end

function tf = iIsAMiniBatchableDatastore(X)
tf = isa(X,'matlab.io.Datastore') && isa(X, 'matlab.io.datastore.MiniBatchable');
end

function tf = iIsValidImageArray(x)
% iIsValidImageArray   Return true if x is an array of
% one or multiple (colour or grayscale) images
tf = iIsRealNumericData( x ) && ...
    ( iIsGrayscale( x ) || iIsColour( x ) || iIsMultiChannel( x )) && ...
    iIs4DArray( x );
end

function tf = iIsValidImage(x)
% iIsValidImage   Return true if x is a non-empty (colour or grayscale) image
tf = ~isempty( x ) && iIsRealNumericData( x ) && ...
    ( iIsGrayscale( x ) || iIsColour( x ) || iIsMultiChannel( x ) ) && ...
    ndims( x ) < 4 ;
end

function tf = iIsValidPath(x)
% iIsValidPath   Return true if x is a valid path. For the moment, we just
% assume any char vector is a valid path.
tf = ischar(x);
end

function tf = iIsValidResponseArray(x)
% iIsValidResponseArray   Return true if x is a vector, a matrix or an
% array of real responses and it does not contain NaNs.
tf = iIsRealNumericData( x ) && ...
    ( isvector(x) || ismatrix(x) || iIs4DArray( x ) ) && ...
    ~iContainsNaNs( x );
end

function tf = iIsRealNumericData(x)
tf = isreal(x) && isnumeric(x);
end

function tf = iIsRealNumericVector(x)
tf = iIsRealNumericData(x) && isvector(x);
end

function tf = iContainsNaNs(x)
if isnumeric(x)
    tf = any(isnan(x(:)));
else
    % If x is not numeric, it cannot contain NaNs since NaN is numeric
    tf = false;
end
end

function tf = iIsGrayscale(x)
tf = size(x,3)==1;
end

function tf = iIsColour(x)
tf = size(x,3)==3;
end

function tf = iIsMultiChannel(x)
tf = size(x,3) > 1;
end

function tf = iIs3DArray(x)
tf = ndims( x ) <= 3;
end

function tf = iIs4DArray(x)
tf = ndims( x ) <= 4;
end

function tf = iHasValidPredictorColumn(tbl)
% iHasValidPredictorColumn   Return true if tbl has a valid predictor
% column as first column. To be valid, the first column should be a cell
% array containing only paths or only image data.
tf = iFirstColumnIsCell(tbl) && ...
    ( iFirstColumnContainsOnlyPaths(tbl) || iFirstColumnContainsOnlyImages(tbl) );
end

function tf = iFirstColumnIsCell(tbl)
% iFirstColumnIsCell   Return true if the first column of tbl is a cell
% array.
tf = iscell(tbl{:,1});
end

function tf = iFirstColumnContainsOnlyPaths(tbl)
% iFirstColumnContainsOnlyPaths   Return true if the first column of tbl
% contains paths and only paths. We do not check if those paths exist,
% since that might be too time consuming.
res = cellfun(@iIsValidPath, tbl{:,1});
tf = all(res);
end

function tf = iFirstColumnContainsOnlyImages(tbl)
% iFirstColumnContainsOnlyImages   Return true if the first column of tbl
% contains images.
res = cellfun(@iIsValidImage, tbl{:,1});
tf = all(res);
end

function tf = iHasValidSeq2OneResponseColumn(tbl)
% iHasValidSeq2OneResponseColumn   Return true if tbl has a valid seq2one
% response column as second column. To be valid, the column should contain
% in-memory data.
tf = isnumeric( tbl{:,2} ) || iscategorical( tbl{:,2} );
end

function tf = iHasValidSeq2SeqResponseColumn(tbl)
% iHasValidSeq2SeqResponseColumn   Return true if tbl has a valid seq2seq
% response column as second column. To be valid, the column should be a
% cell array containing only paths.
tf = iSecondColumnContainsOnlyPaths(tbl);
end

function tf = iSecondColumnContainsOnlyPaths(tbl)
% iSecondColumnContainsOnlyPaths   Return true if the second column of tbl
% contains paths and only paths. We do not check if those paths exist,
% since that might be too time consuming.
tf = false;
if iscell( tbl{:,2})
    res = cellfun(@iIsValidPath, tbl{:,2});
    tf = all(res);
end
end

function tf = iIsACellOf3DNumericArray(x)
tf = iscell(x) && isnumeric(x{:}) && iIs3DArray(x{:});
end

function tf = iHasValidClassificationResponses(tbl)
% iHasValidClassificationResponses   Return true if tbl contains only one
% response column (the second), and responses are stored as a categorical
% vector.
numResponses = size(tbl,2) - 1;
if numResponses ~= 1
    % The table has an incorrect number of response columns
    tf = false;
elseif iscell( tbl{:,2} ) 
    % The table contains file paths of categorical predictors
    tf = iSecondColumnContainsOnlyPaths(tbl);
else
    responses = tbl{:,2};
    responsesAreCategorical = iscategorical(responses);
    % We check the size of the second dimension instead of using isvector
    % to avoid considering horizontal vectors in case there is only one
    % response
    isAVectorOfResponses = size(responses,2)==1;
    tf = isAVectorOfResponses && responsesAreCategorical;
end
end

function tf = iHasValidRegressionResponses(tbl)
% iHasValidRegressionResponses   Return true if tbl contains real scalar
% responses in all columns except the first, or if responses are held in
% one column only in the form of either vectors or cell arrays containing
% 3-D arrays. Responses cannot contain NaNs.
numResponses = width(tbl) - 1;
if numResponses < 1
    % The table has no response column
    tf = false;
elseif numResponses == 1
    % The table has one response column: responses can be scalars, vectors
    % or cell containing 3-D arrays
    scalarOrVectorResponses = all( rowfun(@(x)iIsRealNumericVector(x), tbl(:,2), 'OutputFormat', 'uniform') );
    cellOf3DArrayResponses = all( rowfun(@(x)iIsACellOf3DNumericArray(x), tbl(:,2), 'OutputFormat', 'uniform') );
    containsNaNs = iColumnContainsNaNs(tbl{:,2});
    hasEmptyCells = iColumnContainsEmptyValues(tbl{:,2});
    containsFilePaths = iSecondColumnContainsOnlyPaths(tbl);
    tf = scalarOrVectorResponses || cellOf3DArrayResponses || containsFilePaths;
    tf = tf && ~containsNaNs && ~hasEmptyCells;
else
    % There are multiple response columns: responses can only be scalars
    isRealNumericColumnWithoutNaNsFcn = @(x)(iIsRealNumericVector(x) && ~iColumnContainsNaNs(x));
    tfOnEachColumn = varfun(isRealNumericColumnWithoutNaNsFcn, tbl(:,2:end), 'OutputFormat', 'uniform');
    tf = all( tfOnEachColumn );
end
end

function tf = iColumnContainsNaNs( x )
if iscell( x )
    tf = any(cellfun(@iContainsNaNs, x));
else
    tf = iContainsNaNs( x );
end
end

function tf = iColumnContainsEmptyValues( x )
if iscell( x )
    tf = any(cellfun(@isempty, x));
else
    tf = any(isempty(x));
end
end

function tf = iIsScalarResponseSize( x )
% iIsScalarResponse   Return true if the first two dimensions of the 3-D
% response size x are ones, or if x is a scalar.
if isscalar(x)
    tf = true;
else
    tf = x(1)==1 && x(2)==1;
end
end

function tf = iIsValidSequenceInput( x )
% iIsValidSequenceInput   Return true for cell arrays with real, numeric
% entries, or a single sequence represented by a numeric matrix
validCell = iscell(x) && isvector(x) && all( cellfun(@(s)isreal(s) && ~isempty(s), x) );
validMatrix = isreal(x) && ismatrix(x) && ~isempty(x);
tf = validCell || validMatrix;
end

function tf = iIsValidSequenceCategoricalResponse( x )
% iIsValidSequenceCategoricalResponse   Return true for valid categorical
% or cell array categorical responses
validSeq2OneResponse = iscategorical(x) && isvector(x);
validSeq2SeqResponse = iscell(x) && isvector(x) && all( cellfun(@(s)iscategorical(s) && isrow(s), x) ) ...
    || (iscategorical(x) && isrow(x));
tf = validSeq2OneResponse || validSeq2SeqResponse;
end

function tf = iIsValidSequenceRegressionResponse( x )
% iIsValidSequenceRegressionResponse   Return true for valid sequence
% regression responses
validSeq2OneResponse = isreal(x) && isnumeric(x) && ismatrix(x);
validSeq2SeqResponse = iscell(x) && isvector(x) && all( cellfun(@(s)isreal(s), x) );
tf = validSeq2OneResponse || validSeq2SeqResponse;
end

function tf = iHasDefinedSequenceLabels( x )
% iHasDefinedSequenceLabels   Return true if sequence response has no
% undefined labels
validSeq2OneResponse = iscategorical(x) && all( ~isundefined(x) );
validSeq2SeqResponse = iscell(x) && all( cellfun(@(s)all( ~isundefined(s) ), x) );
tf = validSeq2OneResponse || validSeq2SeqResponse;
end

function tf = iSequencesHaveConsistentDataDimension( x )
% iSequencesHaveConsistentDataDimension   Return true if all sequences
% within a cell array have the same data dimension
if iscell(x)
    firstSize = size( x{1}, 1 );
    tf = all( cellfun( @(s)isequal( size(s, 1), firstSize ), x ) );
else
    % If the sequences are not in a cell, then there is only one
    % observation and dimensions are consistent by default
    tf = true;
end
end

function tf = iSequencesHaveSameNumberOfObservations(x, y)
% iSequencesHaveSameNumberOfObservations   Return true if sequence
% predictors and responses have the same number of observations
if iscell(x) && iscategorical(y)
    tf = numel(x) == numel(y);
elseif iscell(x) && isnumeric(y)
    tf = numel(x) == size(y, 1);
else
    % If the predictors are not in a cell, then there should only be one
    % observation
    tf = ismatrix(x) && (iscategorical(y) || ismatrix(y));
end
end

function tf = iSequencesHaveConsistentSequenceLength( x, y )
% iSequencesHaveConsistentSequenceLength   Return true if sequence
% responses have the same number of timesteps as the corresponding
% predictors
if iscell(y)
    sx = cellfun( @(s)size(s, 2), x );
    sy = cellfun( @(s)size(s, 2), y );
    tf = all( sx == sy );
elseif isnumeric(x) && (iscategorical(y) || isnumeric(y))
    % Assume single observation seq-to-seq case
    tf = size(x, 2) == size(y, 2);
else
    % Otherwise sequences are just labels, so then this is true by default 
    tf = true;
end
end

function tf = iOutputModeMatchesDataForClassification(x, y, returnsSequence)
validSeq2Seq = returnsSequence && ((iscell(x) && iscell(y)) || (isnumeric(x) && iscategorical(y)));
validSeq2One = ~returnsSequence && (iscell(x) && iscategorical(y) && iscolumn(y));
tf = validSeq2One || validSeq2Seq;
end

function tf = iOutputModeMatchesDataForRegression(x, y, returnsSequence)
validSeq2Seq = returnsSequence && ((iscell(x) && iscell(y)) || (isreal(x) && isreal(y)));
validSeq2One = ~returnsSequence && ((iscell(x) && isreal(y) && ismatrix(y)) || (isreal(x) && isreal(y) && ismatrix(y)));
tf = validSeq2One || validSeq2Seq;
end

function tf = iIsScalarResponseOrSeq2OneRNN( networkOutputSize, dataResponseSize, internalLayers)
isRNN = iIsRNN( internalLayers );
if isRNN
    tf = ~iReturnsSequence( internalLayers, isRNN );
else
    tf = iIsScalarResponseSize( networkOutputSize ) && iIsScalarResponseSize( dataResponseSize );
end
end

%% Generic helpers
function arraySize = iArrayResponseNumObservations(y)
% iArrayResponseNumObservations   Return the number of observations of the
% response array y. The number of observations will be the number of
% elements for a categorical vector, the first dimension for a matrix and the last
% dimension when the responses are stored in a 4-D numeric array.
if (iscategorical(y) && isvector( y ))
    arraySize = numel( y );
elseif ismatrix( y )
    arraySize = size( y, 1);
else
    arraySize = size( y, 4 );
end
end

function sizeString = i3DSizeToString( sizeVector )
% i3DSizeToString   Convert a 3-D size stored in a vector of 3 elements
% into a string separated by 'x'.
sizeString = [ ...
    int2str( sizeVector(1) ) ...
    'x' ...
    int2str( sizeVector(2) ) ...
    'x' ...
    int2str( sizeVector(3) ) ];
end

function outputSize = iNetworkOutputSize(internalLayers)
% Determine the output size of the network given the internal layers
inputSize = internalLayers{1}.InputSize;
for i = 2:numel(internalLayers)
    inputSize = internalLayers{i}.forwardPropagateSize(inputSize);
end
outputSize = inputSize;
end

function outputSize = iDAGNetworkOutputSize(lgraph)
% Determine the output size of the network given the layer graph
sizes = extractSizes(lgraph);
outputSize = sizes{end};
end

function internalLayers = iGetInternalLayers( layers )
internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers( layers );
end

function data = iReadDataFromStruct(S)
% Read data from first field in struct S
fn = fieldnames( S );
data = S.(fn{1});
end