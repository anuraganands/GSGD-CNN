classdef FilePathTableMiniBatchDatastore <...
        matlab.io.Datastore &...
        matlab.io.datastore.MiniBatchable &...
        matlab.io.datastore.Shuffleable &...
        matlab.io.datastore.BackgroundDispatchable &...
        matlab.io.datastore.PartitionableByIndex &...
        matlab.io.datastore.internal.ResponseNameable &...
        matlab.io.datastore.internal.FourDArrayReadable
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Dependent)
        MiniBatchSize
    end
    
    properties
        ResponseNames
    end
    
    properties(SetAccess = protected, Dependent)
        NumObservations
    end
    
    properties (Access = private)
        TableData
        ImageDatastore
    end
    
    properties (Access = private)
        StartIndexOfCurrentMiniBatch
        OrderedIndices % Shuffled sequence of all indices into observations
        MiniBatchSizeInternal
    end
    
    methods
        
        function self = FilePathTableMiniBatchDatastore(tableIn,miniBatchSize)
            self.TableData = tableIn;
            self.OrderedIndices = 1:self.NumObservations;
            self.MiniBatchSize = miniBatchSize;
            self.DispatchInBackground = false;
            self.StartIndexOfCurrentMiniBatch = 1;

            if ~isempty(tableIn)
                self.ImageDatastore = iCreateDatastoreFromTable(tableIn);
                self.ResponseNames = iResponseNamesFromTable(self.TableData);
                self.reset();
            end
        end
        
        function [data,info] = readByIndex(self,indices)
            
            info.Response = self.readResponses(self.OrderedIndices(indices));
            
            % Create datastore partition via a copy and index. This is
            % faster than constructing a new datastore with the new
            % files.
            subds = copy(self.ImageDatastore);
            subds.Files = self.ImageDatastore.Files(indices);
            data = subds.readall();
            
        end
        
        function [data,info] = read(self)
            % Map the indices into data
            miniBatchIndices = self.computeDataIndices();
            
            % Read the data
            [data,info.Response] = self.readData(miniBatchIndices);
            
            % Advance indices of current mini batch
            self.advanceCurrentMiniBatchIndices();
        end
        
        function reset(self)
            % Reset iterator state to first mini-batch
            
            self.StartIndexOfCurrentMiniBatch = 1;
            self.ImageDatastore.reset();
        end
        
        function newds = shuffle(self)
            % Shuffle  Shuffle the data
            
            newds = copy(self);
            newds.OrderedIndices = randperm(self.NumObservations);
            newds.ImageDatastore = iCreateDatastoreFromTable(newds.TableData,newds.OrderedIndices);
            newds.MiniBatchSize = self.MiniBatchSize;
        end
        
        function reorder(self,indices)
            self.OrderedIndices = indices;
            currentMiniBatchSize = self.MiniBatchSize;
            self.Datastore = iCreateDatastoreFromTable(self.TableData,self.OrderedIndices);
            self.MiniBatchSize = currentMiniBatchSize;
        end
        
        function TF = hasdata(self)
            TF = self.StartIndexOfCurrentMiniBatch <= self.NumObservations;
        end
        
        function newds = partitionByIndex(self,indices)
            partitionTable = self.TableData(self.OrderedIndices(indices),:);
            newds = nnet.internal.cnn.FilePathTableMiniBatchDatastore(partitionTable,self.MiniBatchSize);
        end
        
        function numObs = get.NumObservations(self)
            if isempty(self.TableData)
                numObs = 0;
            else
                numObs = size(self.TableData,1);
            end
        end
        
        function set.MiniBatchSize(self,batchSize)
            self.ImageDatastore.ReadSize = batchSize;
        end
        
        function batchSize = get.MiniBatchSize(self)
            batchSize = self.ImageDatastore.ReadSize;
        end
        
    end
    
    methods (Hidden)
       
        function frac = progress(self)
            if hasdata(self)
                frac = (self.StartIndexOfCurrentMiniBatch - 1) / self.NumObservations;
            else
                frac = 1;
            end            
        end
        
    end
    
    methods (Access = private)
        
        function [X,Y] = readData(self,indices)
            if isempty(self.TableData)
                [X,Y] = deal([]);
            else
                X = self.ImageDatastore.read();
                Y = self.readResponses(indices);
            end
        end
        
        function response = readResponses(self,indices)
            
            singleResponseColumn = size(self.TableData,2) == 2;
            if singleResponseColumn
                response = self.TableData{indices,2};
                if isvector(self.TableData(1,2))
                    response = iMatrix2Tensor(response);
                end
            else
                response = iMatrix2Tensor(self.TableData{indices,2:end});
            end
            
        end
        
        function dataIndices = computeDataIndices(self)
            % computeDataIndices    Compute the indices into the data from
            % start and end index
            startIdx = min(self.StartIndexOfCurrentMiniBatch,self.NumObservations);
            endIdx = startIdx + self.MiniBatchSize - 1;
            endIdx = min(endIdx,self.NumObservations);
            
            dataIndices = startIdx:endIdx;
            
            % Convert sequential indices to ordered (possibly shuffled) indices
            dataIndices = self.OrderedIndices(dataIndices);
        end
        
        function advanceCurrentMiniBatchIndices(self)
            self.StartIndexOfCurrentMiniBatch = self.StartIndexOfCurrentMiniBatch + self.MiniBatchSize;
        end
        
    end
end

function dataStore = iCreateDatastoreFromTable( aTable, shuffleIdx )

% Assume the first column of the table contains the paths to the images
if nargin < 2
    filePaths = aTable{:,1}'; % 1:end
else
    filePaths = aTable{shuffleIdx,1}'; % Specific shuffle order
end

if any( cellfun(@isdir,filePaths) )
    % Directories are not valid paths
    iThrowWrongImagePathException();
end
try
    dataStore = imageDatastore( filePaths );
catch e
    iThrowFileNotFoundAsWrongImagePathException(e);
    iThrowInvalidStrAsEmptyPathException(e);
    rethrow(e)
end
numObservations = size( aTable, 1 );
numFiles = numel( dataStore.Files );
if numFiles ~= numObservations
    % If some files were discarded when the datastore was created, those
    % files were not valid images and we should error out
    iThrowWrongImagePathException();
end
end

function iThrowWrongImagePathException()
% iThrowWrongImagePathException   Throw a wrong image path exception
exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:TableMiniBatchDatastore:WrongImagePath');
throwAsCaller(exception)
end

function iThrowFileNotFoundAsWrongImagePathException(e)
% iThrowWrongImagePathException   Throw a
% MATLAB:datastoreio:pathlookup:fileNotFound as a wrong image path
% exception.
if strcmp(e.identifier,'MATLAB:datastoreio:pathlookup:fileNotFound')
    iThrowWrongImagePathException()
end
end

function iThrowInvalidStrAsEmptyPathException(e)
% iThrowInvalidStrAsEmptyPathException   Throws a
% pathlookup:invalidStrOrCellStr exception as a EmptyImagePaths exception
if (strcmp(e.identifier,'MATLAB:datastoreio:pathlookup:invalidStrOrCellStr'))
    exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:TableMiniBatchDatastore:EmptyImagePaths');
    throwAsCaller(exception)
end
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(errorID, getString(message(errorID, varargin{:})));
end

function tensorResponses = iMatrix2Tensor( matrixResponses )
% iMatrix2Tensor   Convert a matrix of responses of size numObservations x
% numResponses to a tensor of size 1 x 1 x numResponses x numObservations
if iscategorical( matrixResponses )
    tensorResponses = matrixResponses;
else
    [numObservations, numResponses] = size( matrixResponses );
    tensorResponses = matrixResponses';
    tensorResponses = reshape(tensorResponses,[1 1 numResponses numObservations]);
end
end

function responseNames = iResponseNamesFromTable( tableData )
if size(tableData,2) > 1
    responseNames = tableData.Properties.VariableNames(2:end);
    % To be consistent with ClassNames, return a column array
    responseNames = responseNames';
else
    responseNames = {};
end
end
