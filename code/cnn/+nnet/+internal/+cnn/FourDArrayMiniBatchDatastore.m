classdef FourDArrayMiniBatchDatastore < ...
        matlab.io.Datastore &...
        matlab.io.datastore.MiniBatchable &...
        matlab.io.datastore.Shuffleable &...
        matlab.io.datastore.BackgroundDispatchable &...
        matlab.io.datastore.PartitionableByIndex &...
        matlab.io.datastore.internal.FourDArrayReadable
    
    %   Copyright 2017 The MathWorks, Inc.
    
    % Required Datastore interface
    methods
        
        function self = FourDArrayMiniBatchDatastore(X,Y,miniBatchSize)
            
            self.MiniBatchSize = miniBatchSize;
            self.StartIndexOfCurrentMiniBatch = 1;
            self.DispatchInBackground = false;
            
            if ~isempty(X)
                iValidateNumericInputs(X,Y);
                self.Input = X;
                self.OrderedIndices = 1:self.NumObservations;
                
                if ~isempty(Y)
                    self.Response = Y;
                end
            end
        end
        
        function [data,info] = read(self)
            
            % Get current observation indices
            miniBatchIndices = self.computeDataIndices();
            
            % Read the data
            [data,Y] = self.readData(miniBatchIndices);
            info.MiniBatchIndices = miniBatchIndices;
            
            % Package X and Y cell arrays in table
            info.Response = Y;
            
            % Advance indices of current mini batch
            self.advanceCurrentMiniBatchIndices();
            
        end
        
        function [data,info] = readByIndex(self,indices)
            
            [data,Y] = self.readData(self.OrderedIndices(indices));
            info.MiniBatchIndices = self.OrderedIndices(indices);
            
            % Package Response in info struct
            info.Response = Y;
            
        end
                
        function subds = partitionByIndex(ds,idx)
            
            indices = ds.OrderedIndices(idx);
            inputPartition = ds.Input(:, :, :, indices);
            % If response is a vector or 2D array, it should be partitioned
            % along its rows. If it's a 4D array, it should be partitioned
            % along dim 4.
            if ~isempty(ds.Response)
                if ismatrix(ds.Response)
                    responsePartition = ds.Response(indices, :);
                else
                    responsePartition = ds.Response(:, :, :, indices);
                end
            else
                responsePartition = [];
            end
            subds = nnet.internal.cnn.FourDArrayMiniBatchDatastore(inputPartition,responsePartition,ds.MiniBatchSize);
        end
        
        function dsnew = shuffle(self)
            dsnew = copy(self);
            dsnew.OrderedIndices = randperm(self.NumObservations);
        end
        
        function reset(self)
            self.StartIndexOfCurrentMiniBatch = 1;
        end
        
        function TF = hasdata(self)
            TF = self.StartIndexOfCurrentMiniBatch <= self.NumObservations;
        end
        
        function numObs = get.NumObservations(self)
            if isempty(self.Input)
                numObs = 0;
            else
                numObs = size(self.Input,4);
            end
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
            X = readInput(self,indices);
            Y = readResponses(self,indices);
        end
        
        function inputs = readInput(self, indices)
            inputs = self.Input(:,:,:,indices);
        end
        
        function responses = readResponses(self, indices)
            if isempty(self.Response)
                responses = [];
            else
                if iscategorical(self.Response)
                    % Categorical vector of responses
                    responses = self.Response(indices);
                    responses = reshape(responses,[length(indices),1]);
                elseif ismatrix(self.Response)
                    % Matrix of responses
                    responses = iMatrix2Tensor(self.Response(indices,:));
                else
                    % 4D array of responses already in the right shape
                    responses = self.Response(:,:,:,indices);
                end
                
                if isnumeric(responses)
                    responsesCell = cell(self.MiniBatchSize,1);
                    for idx = 1:self.MiniBatchSize
                        responsesCell{idx} = responses(:,:,:,idx);
                    end
                    responses = responsesCell;
                end
                
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
    
    properties (Access = public)
        MiniBatchSize
    end
    
    properties (SetAccess = protected,Dependent)
        NumObservations
    end
    
    properties (Access = private)
        
        Input
        Response
        StartIndexOfCurrentMiniBatch
        OrderedIndices
        
    end
    
end


function tensorResponses = iMatrix2Tensor( matrixResponses )
% iMatrix2Tensor   Convert a matrix of responses of size numobservations x
% numResponses to a tensor of size 1 x 1 x numResponses x numObservations
[numObservations, numResponses] = size( matrixResponses );
tensorResponses = matrixResponses';
tensorResponses = reshape(tensorResponses,[1 1 numResponses numObservations]);
end

function iValidateNumericInputs(input,response)

if isempty(response)
    return % For inference use cases.
end

if iscategorical(response)
    if size(input,4) ~= length(response)
        exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:FourDArrayMiniBatchDatastore:XandYNumObservationsDisagree');
        throwAsCaller(exception);
    end
elseif isnumeric(response)
    if ismatrix(response) && (size(response,1) ~= size(input,4))
        exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:FourDArrayMiniBatchDatastore:XandYNumObservationsDisagree');
        throwAsCaller(exception);
    elseif (ndims(response) == 4) && (size(response,4) ~= size(input,4))
        exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:FourDArrayMiniBatchDatastore:XandYNumObservationsDisagree');
        throwAsCaller(exception);
    elseif (ndims(response) ~= 4) && ~ismatrix(response) && (ndims(response) ~= 3)
        exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:FourDArrayMiniBatchDatastore:YWrongDimensionality');
        throwAsCaller(exception);
    end
else
    exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:FourDArrayMiniBatchDatastore:UnexpectedTypeProvidedForY');
    throwAsCaller(exception);
end

end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(errorID, getString(message(errorID, varargin{:})));
end
