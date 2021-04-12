classdef InMemoryTableMiniBatchDatastore <...
        matlab.io.Datastore &...
        matlab.io.datastore.MiniBatchable &...
        matlab.io.datastore.Shuffleable &...
        matlab.io.datastore.BackgroundDispatchable &...
        matlab.io.datastore.PartitionableByIndex &...
        matlab.io.datastore.internal.ResponseNameable &...
        matlab.io.datastore.internal.FourDArrayReadable
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        MiniBatchSize
    end
    
    properties
        ResponseNames
    end
    
    properties (SetAccess = protected, Dependent)
        NumObservations
    end
    
    properties (Access = private)
        TableData
    end
    
    properties (Access = private)
        StartIndexOfCurrentMiniBatch
        OrderedIndices % Shuffled sequence of all indices into observations
    end
    
    methods
        
        function self = InMemoryTableMiniBatchDatastore(tableIn,miniBatchSize)

            self.TableData = tableIn;
            self.OrderedIndices = 1:self.NumObservations;
            self.MiniBatchSize = miniBatchSize;
            self.DispatchInBackground = false;
            self.StartIndexOfCurrentMiniBatch = 1;

            if ~isempty(tableIn)
                self.ResponseNames = iResponseNamesFromTable(self.TableData);
            end
        end
        
        function numObs = get.NumObservations(self)
            if isempty(self.TableData)
                numObs = 0;
            else
                numObs = size(self.TableData,1);
            end
        end
        
        function [data,info] = read(self)
            % read  Return next mini-batch of data
            
            % Map the indices into data
            miniBatchIndices = self.computeDataIndices();
            
            % Read the data
            [data,Y] = self.readData(miniBatchIndices);
            info.Response = Y;
            
            % Advance indices of current mini batch
            self.advanceCurrentMiniBatchIndices();
        end
        
        function [data,info] = readByIndex(self,indices)
            [data,Y] = self.readData(self.OrderedIndices(indices));
            info.Response = Y;
        end
        
        function reset(self)
            % reset  Reset iterator state to first mini-batch
            self.StartIndexOfCurrentMiniBatch = 1;
        end
        
        function dsnew = shuffle(self)
            % shuffle  Shuffle the data
            dsnew = copy(self);
            dsnew.OrderedIndices = randperm(self.NumObservations);
        end
        
        function reorder(self,indices)
            % reorder   Shuffle the data to a specific order
            self.OrderedIndices = indices;
        end
        
        function dsnew = partitionByIndex(self,indices)
            idx = self.OrderedIndices(indices);
            partitionedTable = self.TableData(idx,:);
            dsnew = nnet.internal.cnn.InMemoryTableMiniBatchDatastore(partitionedTable,self.MiniBatchSize);
        end
        
        function TF = hasdata(self)
            TF = self.StartIndexOfCurrentMiniBatch <= self.NumObservations;
        end
        
    end
    
    methods (Access = private)
        
        function [X,Y] = readData(self,indices)
            if isempty(self.TableData)
                [X,Y] = deal([]);
            else
                X = self.readInput(indices);
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
        
        function X = readInput(self,indices)
            
            X = self.TableData{indices,1};
            if any(cellfun(@isempty,X))
                iThrowEmptyImageDataException();
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
    
    methods (Hidden)
       
        function frac = progress(self)
            if hasdata(self)
                frac = (self.StartIndexOfCurrentMiniBatch - 1) / self.NumObservations;
            else
                frac = 1;
            end            
        end
    end
end

function iThrowEmptyImageDataException()
% iThrowEmptyImageDataException   Throw an empty image data exception
exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:TableMiniBatchDatastore:EmptyImageData');
throwAsCaller(exception)
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
