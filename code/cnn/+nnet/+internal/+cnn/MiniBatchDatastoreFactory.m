% MiniBatchDatastoreFactory   Factory for making MiniBatchDatastores
%
%   mbds = MiniBatchDatastoreFactoryInstance.createMiniBatchDatastore(data)
%   data: the data to be dispatched.
%       According to their type the appropriate MiniBatchDatastore will be used.
%       Supported types: 4-D double, imageDatastore, table.

%   Copyright 2017 The MathWorks, Inc.

classdef MiniBatchDatastoreFactory
     
    methods (Static)
        function mbds = createMiniBatchDatastore( inputs, response, initMiniBatchSize)
            % createMiniBatchDatastore   Create MiniBatchDatastore
            %
            % Syntax:
            %     createMiniBatchDatastore(inputs, response)
            
            if nargin < 3
                initMiniBatchSize = 128;
            end
            
            if iIsRealNumeric4DHostArray(inputs)
                 mbds = nnet.internal.cnn.FourDArrayMiniBatchDatastore(inputs, response, initMiniBatchSize);
            elseif isa(inputs, 'matlab.io.datastore.ImageDatastore')
                mbds  = nnet.internal.cnn.ImageDatastoreMiniBatchDatastore(inputs, initMiniBatchSize);
            elseif istable(inputs)
                if iIsAnInMemoryTable(inputs)
                    mbds = nnet.internal.cnn.InMemoryTableMiniBatchDatastore(inputs, initMiniBatchSize);
                else
                    mbds = nnet.internal.cnn.FilePathTableMiniBatchDatastore(inputs, initMiniBatchSize);
                end
            elseif isa(inputs,'matlab.io.Datastore') && isa(inputs,'matlab.io.datastore.MiniBatchable')
                % If passed MiniBatchable Datastore, use in layered
                % composition.
                mbds = inputs;
            else
               error( message( 'nnet_cnn:internal:cnn:MiniBatchDatastoreFactory:InvalidData' ) );
            end
           
        end
        
    end
    
end

function tf = iIsAnInMemoryTable( x )
if isempty(x)
    % If given empty table, allow to pass through. Doesn't matter whether
    % we consider this an InMemory empty table or a filepath empty table,
    % won't dispatch either way because it is empty.
    tf = true;
else
    firstCell = x{1,1};
    tf = isnumeric( firstCell{:} );
end
end

function tf = iIsRealNumeric4DHostArray( x )
tf = iIsRealNumericData( x ) && iIsValidImageArray( x ) && ~iIsGPUArray( x );
end

function tf = iIsRealNumericData(x)
tf = isreal(x) && isnumeric(x) && ~issparse(x);
end

function tf = iIsValidImageArray(x)
% iIsValidImageArray   Return true if x is an array of
% one or multiple (colour, grayscale or Multi-Channel) images
tf = ( iIsGrayscale( x ) || iIsColour( x ) || iIsMultiChannel( x ) ) && ...
    iIs4DArray( x );
end

function tf = iIsGrayscale(x)
tf = size(x,3)==1;
end

function tf = iIsColour(x)
tf = size(x,3)==3;
end

function tf = iIsMultiChannel(x)
tf = size(x,3)>1;
end

function tf = iIs4DArray(x)
tf = ndims(x) <= 4;
end

function tf = iIsGPUArray( x )
tf = isa(x, 'gpuArray');
end