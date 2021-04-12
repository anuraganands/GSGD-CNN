classdef( Hidden, Sealed ) BackgroundDispatcher < nnet.internal.cnn.DataDispatcher
% BackgroundDispatcher  Dispatcher wrapper to move dispatch into the
% background on a parallel pool
    
    %   Copyright 2017 The MathWorks, Inc.
    
    
    properties (SetAccess = private)
        % ImageSize  (1x3 int) Size of each image to be dispatched
        ImageSize
        
        % ResponseSize   (1x3 int) Size of each response to be dispatched
        ResponseSize
        
        % NumObservations   (int) Number of observations in the data set
        NumObservations
        
        % IsDone (logical)  Flags when the data has been used up
        IsDone
        
        % ClassNames (cellstr) Array of class names corresponding to
        %            training data labels.
        ClassNames
        
        % ResponseNames (cellstr) Array of response names corresponding to
        %               training data response names.
        ResponseNames
    end
    
    properties
        % MiniBatchSize (int)   Number of elements in a mini batch
        MiniBatchSize
        
        % Precision   Precision used for dispatched data
        Precision
        
        % EndOfEpoch    Strategy to choose how to cope with a number of
        % observations that is not divisible by the desired number of mini
        % batches
        %
        % Allowed values: 'truncateLast', 'discardLast'
        EndOfEpoch
    end
    
    properties( Access = private )
        % Dispatcher  The contained dispatcher object
        Dispatcher
        
        % CurrentMiniBatch  Keeps track of the current mini-batch as we
        % progress through the data
        CurrentMiniBatch = 0
        
        % NextMiniBatch  Keeps track of the next mini-batch as we
        % progress through the data
        NextMiniBatch = 0
        
        % NumMiniBatches  How many mini-batches are there for the given
        % MiniBatchSize
        NumMiniBatches
        
        % IsInitialized  Flags whether dispatchers have been copied to the
        % pool and are ready to be used
        IsInitialized = false
        
        % QueueLength  How many jobs should be running fetching future
        % mini-batches
        QueueLength
        
        % BackgroundReadQueue  Queue of PARFEVAL futures running next() on
        % background workers to get mini-batches of data
        BackgroundReadQueue = parallel.FevalFuture.empty
        
        % BackgroundDispatchers  Copies of the stored dispatcher passed to
        % the background workers as parallel.pool.Constant objects
        BackgroundDispatchers
    end
    
    properties( Constant )
        % QueueLengthFactor   How many more background jobs should be
        % running than there are pool workers? Oversubscription ensures
        % all workers are always busy, but if they all finish early the
        % client is storing all the results, which has a memory impact.
        QueueLengthFactor = 2
    end
    
    methods
        
        function this = BackgroundDispatcher( dispatcher )
        % BackgroundDispatcher  Construct BackgroundDispatcher containing
        % the given dispatcher
        
            % If the dispatcher already is a BackgroundDispatcher, do not
            % try to wrap it again
            assert( ~isa( dispatcher, 'nnet.internal.cnn.BackgroundDispatcher' ), ...
                'Tried to wrap a BackgroundDispatcher with another BackgroundDispatcher' );
            
            % Check that this environment and dispatcher supports
            % background dispatch
            nnet.internal.cnn.BackgroundDispatcher.checkCanRunInBackground( dispatcher );
            
            % Copy the dispatcher and its various properties so that this
            % object appears the same to any users
            this.Dispatcher = dispatcher;
            this.NumObservations = dispatcher.NumObservations;
            this.ClassNames = dispatcher.ClassNames;
            this.ResponseNames = dispatcher.ResponseNames;
            this.MiniBatchSize = dispatcher.MiniBatchSize;
            this.EndOfEpoch = dispatcher.EndOfEpoch;
            this.Precision = dispatcher.Precision;
            this.ImageSize = dispatcher.ImageSize;
            this.ResponseSize = dispatcher.ResponseSize;
            this.IsDone = false;
            
            % Compute how many mini-batches there
            ceilOrFloor = @floor;
            if this.EndOfEpoch == "truncateLast"
                ceilOrFloor = @ceil;
            end
            this.NumMiniBatches = ceilOrFloor(this.NumObservations / this.MiniBatchSize);
            
        end
        
        function delete( this )
        % delete  Makes sure nothing is keeping this object alive when it
        % goes out of scope
            flushQueue( this );
        end
        
    end
    
    methods
        
        function [miniBatchData, miniBatchResponse, miniBatchIndices] = next( this )
        % next   Get the data and response for the next mini batch
            
            if isempty(this.BackgroundReadQueue)
                fillQueue( this );
            end
            
            % Pop a future from the queue and get its outputs
            future = this.BackgroundReadQueue(1);
            % Display any command window output on the client
            wait(future);
            disp(future.Diary);
            % Errors should be thrown here as if by the client
            if ~isempty(future.Error)
                throw(future.Error);
            end
            [miniBatchData, miniBatchResponse, miniBatchIndices] = ...
                future.fetchOutputs();
            this.BackgroundReadQueue(1) = [];
            
            % Move the counters on
            this.CurrentMiniBatch = this.NextMiniBatch;
            this.NextMiniBatch = getNextBatchIndex(this, this.NextMiniBatch);
            if ~this.IsDone && this.CurrentMiniBatch == this.NumMiniBatches
                % Latch IsDone to true until cleared by start()
                this.IsDone = true;
            end
            
            % Ensure queue is full
            fillQueue( this );
            
        end
        
        function start( this )
        % start  Initialize if needed and move read point to start
            
            initialize( this );
            
            % Only flush the queue if the next mini-batch wasn't going to
            % be the first anyway
            if ~(this.NextMiniBatch == 1)
                flushQueue( this );
            end
            this.CurrentMiniBatch = 0;
            this.NextMiniBatch = 1;
            this.IsDone = false;
            
            % Start fetching data
            fillQueue( this );
            
        end
        
        function shuffle( this )
        % shuffle  Mix the observations
                        
            % If not initialized then just shuffle the client's dispatcher
            if ~this.IsInitialized
                shuffle( this.Dispatcher );
            else
                                
                % Flush the queue
                flushQueue( this );
                
                % If the reorder method has been implemented by the
                % contained dispatcher then we can use it to reorder the
                % data without the need to copy a new version of the
                % dispatcher to the pool
                if implementsReorder( this.Dispatcher )
                    newOrder = randperm(this.NumObservations);
                    
                    % Reorder client's copy
                    reorder( this.Dispatcher, newOrder );
                    
                    % Reorder pool copies
                    wait( parfevalOnAll(@(c)reorder(c.Value, newOrder), 0, this.BackgroundDispatchers) );
                else
                    % There is no implementation of reorder, so we have to
                    % shuffle the client's datastore and update the copies
                    % on the pool
                    shuffle( this.Dispatcher );
                    this.BackgroundDispatchers = parallel.pool.Constant( this.Dispatcher );
                end
                
            end
        end
        
    end
    
    methods( Access = private )
        
        function initialize( this )
        % initialize  One-time initialization of this object, creating the
        % pool and copying the dispatchers to it
            
            if ~this.IsInitialized
                
                % Validate the pool or create one if necessary
                pool = gcp('nocreate');
                if ~isempty(pool)
                    % Check that the open pool is local
                    if ~isa( pool.Cluster, 'parallel.cluster.Local' )
                        error(message('nnet_cnn:internal:cnn:BackgroundDispatcher:CurrentPoolNotLocal'));
                    end
                else % Open a pool in the default profile
                    % Check that the default cluster profile is local
                    defaultProfileName = parallel.defaultClusterProfile();
                    defaultProfileType = parallel.internal.settings.ProfileExpander.getClusterType( defaultProfileName );
                    if defaultProfileType == parallel.internal.types.SchedulerType.Local
                        pool = parpool;
                    else
                        error(message('nnet_cnn:internal:cnn:BackgroundDispatcher:DefaultClusterNotLocal', defaultProfileName));
                    end
                end
                % Queue length is a proportion of the pool size
                this.QueueLength = floor(pool.NumWorkers * this.QueueLengthFactor);
                
                % Copy dispatcher to the pool
                this.BackgroundDispatchers = parallel.pool.Constant( this.Dispatcher );
                
                % Put dispatcher into a ready state
                this.CurrentMiniBatch = 0;
                this.NextMiniBatch = 1;
                this.IsInitialized = true;
                
            end
            
        end
        
        function nextBatchIndex = getNextBatchIndex( this, currentBatchIndex )
        % nextBatchIndex   Helper to compute the next batch index assuming
        % wrap-around
            nextBatchIndex = mod( currentBatchIndex, this.NumMiniBatches ) + 1;
        end
        
        function fillQueue( this )
        % fillQueue   Maintain a queue of PARFEVAL jobs fetching the next
        % few mini-batches of data
            
            for q = (numel(this.BackgroundReadQueue) + 1):this.QueueLength
                
                batchIndex = getNextBatchIndex(this, this.CurrentMiniBatch + q - 1);
                this.BackgroundReadQueue(q) = ...
                    parfeval( @iCallMethodOnPoolConstant, 3, @getBatch, ...
                    this.BackgroundDispatchers, batchIndex );
                
            end
            
        end
        
        function flushQueue( this )
        % flushQueue   Empty the queue of PARFEVAL fetch jobs because they
        % are no longer valid
            cancel( this.BackgroundReadQueue );
            this.BackgroundReadQueue = parallel.FevalFuture.empty;
        end
        
    end
    
    methods( Static )
        
        function checkCanRunInBackground( dispatcher )
        % checkCanRunInBackground  Checks whether this environment and this
        % dispatcher allow running in the background
            
            % Check whether PCT is installed and licensed
            if ~nnet.internal.cnn.util.canUsePCT()
                error( message( ...
                    'nnet_cnn:internal:cnn:BackgroundDispatcher:PCTNotInstalled' ) );
            end
            
            % Check that this dispatcher implements the minimum interface
            % needed. Check that it implements the
            % BackgroundCapableDispatcher mixin, but also actually
            % implements the necessary methods.
            if ~(iHasOverload( dispatcher, 'getBatch' ) || ...
                    iHasOverload( dispatcher, 'getObservations' ))
                error( message( ...
                    'nnet_cnn:internal:cnn:BackgroundDispatcher:DispatcherNotSupported', ...
                    class(dispatcher) ) );
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

function varargout = iCallMethodOnPoolConstant( F, constantObject, varargin )
% iCallMethodOnPoolConstant  Helper to call a method on a
% parallel.pool.Constant wrapping an object
[varargout{1:nargout}] = feval( F, constantObject.Value, varargin{:} );
end