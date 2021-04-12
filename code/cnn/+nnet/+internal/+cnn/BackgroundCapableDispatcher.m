classdef (Abstract) BackgroundCapableDispatcher < handle
% BackgroundCapableDispatcher   Mixin for dispatcher that can be run in the
% background
    
%   Copyright 2017 The MathWorks, Inc.

    properties (Access = protected)
        % RunInBackgroundOnAuto  For 'auto' background dispatch,
        % allows each Dispatcher to define preferred behaviour, since
        % running in the background is not always advisable.
        RunInBackgroundOnAuto = true
    end
    
    properties (SetAccess = protected)
        % RunInBackground  Flag to request dispatch to take place in a
        % background process
        RunInBackground = false
    end
    
    methods (Hidden)
        
        function [data, response, indices] = getObservations(this, indices) %#ok<INUSL,STOUT>
        % getObservations  Get a batch of observations as specified by
        % their indices. The base class version asserts and should never be
        % called.
            assert( false, 'Dispatcher has no implementation of getObservations method' );
        end
        
        function [miniBatchData, miniBatchResponse, miniBatchIndices] = getBatch(this, batchIndex)
        % getBatch   Get the data and response for a specific mini batch
        % and corresponding indices. Base class implementation uses
        % getObservations but can be overloaded.
            
            % Work out which observations go with this batch
            miniBatchStartIndex = ( (batchIndex - 1) * this.MiniBatchSize ) + 1;
            miniBatchEndIndex = min( this.NumObservations, miniBatchStartIndex + this.MiniBatchSize - 1 );
            miniBatchIndices = miniBatchStartIndex:miniBatchEndIndex;
            
            % Read the data
            [miniBatchData, miniBatchResponse, miniBatchIndices] = this.getObservations(miniBatchIndices);
        end
        
        function reorder( this, indices ) %#ok<INUSD>
        % reorder  Reorder the observations according to the given indices.
        % The base class version asserts and should never be called.
            assert( false, 'Dispatcher has no implementation of reorder method' );
        end
    
        function tf = setRunInBackground( this, flag )
        % setRunInBackground  Flag whether to request that dispatch take
        % place in a background process. Checks whether PCT is installed
        % and licensed to allow this to happen.
        
            if nargin > 1 && string(flag) ~= "auto"
                this.RunInBackground = flag;
            else
                this.RunInBackground = this.RunInBackgroundOnAuto;
            end
            
            tf = this.RunInBackground;
            
            % Validate that background dispatch is allowed
            if tf
                nnet.internal.cnn.BackgroundDispatcher.checkCanRunInBackground( this );
            end
        end
        
        function TF = implementsReorder( this )
            % implementsReorder  check whether a dispatcher supports the use
            % of the reorder method to optimize shuffling when doing
            % BackgroundDispatch.
            
            TF = iHasOverload( this, 'reorder' );
        end
        
    end
    
    methods (Access = protected)
        
        function checkValidReorderIndices( this, indices )
        % checkValidReorderIndices   Helper for REORDER method
            if ~(isempty(indices) && this.NumObservations == 0)
                if ~isequal( double(sort(indices(:))), (1:this.NumObservations)' )
                    error( message('nnet_cnn:internal:cnn:BackgroundCapableDispatcher:ReorderCannotChangeObservations') );
                end
            end
        end
                
    end
end

function tf = iHasOverload( object, methodName )
% Determines whether or not an object or one of its superclasses (between
% it and the base class) has overloaded a particular method that is
% implemented in the base class
meta = metaclass( object );
% Base class can't really be deduced from the metadata because MATLAB
% allows mixins, so we must name it explicitly
baseClass = 'nnet.internal.cnn.BackgroundCapableDispatcher';

tf = true;
methodList = meta.MethodList;
methodIndex = find( strcmp( {methodList.Name}, methodName ), 1, 'first' );
methodMeta = methodList(methodIndex);
if isempty(methodMeta) || isequal(methodMeta.DefiningClass.Name, baseClass)
    tf = false;
end
end
