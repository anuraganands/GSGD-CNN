classdef DistributedDispatcher < nnet.internal.cnn.DataDispatcher
    % DistributedDispatcher Dispatcher split onto workers of a parallel
    % pool.
    %
    % Objects of this class reside on the CLIENT, only the Composite
    % property DistributedData can actually be passed into the pool.
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % ImageSize  (1x3 int) Size of each image to be dispatched
        ImageSize
        
        % ResponseSize   (1x3 int) Size of each response to be dispatched
        ResponseSize
        
        % NumObservations   (int) Number of observations in the data set
        NumObservations
        
        % IsDone (logical)  Unneeded Abstract property that must be
        % overloaded
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
    
    properties (SetAccess = private)
        % Proportions (double)  Proportion of data on each worker
        Proportions
        
        % SubBatchSizes (double)  Size of the portion of a minibatch
        % processed on each worker (sums to MiniBatchSize)
        SubBatchSizes
    end
    
    properties (SetAccess = private)
        % DistributedData (Composite) Dispatcher on each worker, each with
        % a partition of the data
        DistributedData
    end
    
    properties (Access = private)
        IsInitialized = false
    end
    
    methods
        function this = DistributedDispatcher(data, workerLoad, preserveDataOrder)
            % DistributedDispatcher   Constructor for a distributed
            % dispatcher based on an input data dispatcher
            %
            % data            - The DataDispatcher to be distributed
            % workerLoad      - Proportions of the data to be processed
            %                 by each worker, array of length numlabs.
            %                 Elements can be zero, meaning this worker
            %                 does nothing
            % preserveDataOrder - Ensure that the order that data will be
            %                 read is unchanged by distribution
            if ~isa( data, 'nnet.internal.cnn.DistributableDispatcher' )
                error(message('nnet_cnn:internal:cnn:DistributedDispatcher:DispatcherNotDistributable', class( data )));
            end
            if preserveDataOrder && ~data.CanPreserveOrder
                error(message('nnet_cnn:internal:cnn:DistributedDispatcher:DispatcherCannotPreserveOrder', class( data )));
            end
            
            this.NumObservations = data.NumObservations;
            this.ClassNames = data.ClassNames;
            this.ResponseNames = data.ResponseNames;
            this.MiniBatchSize = data.MiniBatchSize;
            this.EndOfEpoch = data.EndOfEpoch;
            this.Precision = data.Precision;
            this.ImageSize = data.ImageSize;
            this.ResponseSize = data.ResponseSize;
            
            this.Proportions = iCalculateProportions( workerLoad );
            
            % The best way to ensure zero bias towards any one grouping of
            % similar observations on any one worker is to shuffle the data
            % before distribution.
            if ~preserveDataOrder
                data.shuffle();
            end
            
            % Distribute data to workers in the specified proportions
            [distributedDataCell, this.SubBatchSizes] = distribute( data, this.Proportions );
            distributedData = Composite();
            
            % If we're doing dispatch in the background then wrap the
            % dispatcher partitions on each worker in an appropriate
            % background or compute server
            if isa( data, 'nnet.internal.cnn.BackgroundCapableDispatcher' ) && ...
                    data.RunInBackground

                % Wrap on the workers
                spmd
                    distributedData = ...
                        nnet.internal.cnn.ParallelBackgroundAdapter.createParallelBackgroundDispatch( ...
                        distributedDataCell, this.Proportions, this.NumObservations, this.MiniBatchSize, this.EndOfEpoch );
                end
            else
                % Wrap on the workers
                [ distributedData{:} ] = deal( distributedDataCell{:} );
                spmd
                    distributedData = nnet.internal.cnn.SimpleRemoteDispatcher( ...
                        distributedData, this.NumObservations, this.MiniBatchSize, this.EndOfEpoch );
                end
            end
            this.DistributedData = distributedData;
            
            this.IsInitialized = true;
        end
        
        % set.EndOfEpoch  Propagate change of EndOfEpoch to the distributed
        % data
        function set.EndOfEpoch(this, endOfEpoch)
            this.EndOfEpoch = endOfEpoch;
            distributedData = this.DistributedData; %#ok<MCSUP>
            if ~isempty(distributedData)
                spmd
                    distributedData.setEndOfEpoch(endOfEpoch);
                end
            end
        end
        
        % set.Precision  Overload to assert. If setting the Precision after
        % construction is needed then this will need to be propagated to
        % the remote dispatchers
        function set.Precision(this, precision)
            this.Precision = precision;
            if this.IsInitialized %#ok<MCSUP>
                assert(false, 'Setting the Precision on a DistributedDispatcher after construction is not currently supported');
            end
        end
        
        % set.MiniBatchSize Overload to assert. If setting the
        % MiniBatchSize after construction is needed then this will need to
        % be propagated to the remote dispatchers
        function set.MiniBatchSize(this, precision)
            this.MiniBatchSize = precision;
            if this.IsInitialized %#ok<MCSUP>
                assert(false, 'Setting the MiniBatchSize on a DistributedDispatcher after construction is not currently supported');
            end
        end
        
        % next  Declare but do not implement, not meaningful to call on the
        % client
        next(~)
        
        % start  Declare but do not implement, not meaningful to call on the
        % client
        start(~)
        
        % shuffle  Declare but do not implement, not meaningful to call on
        % the client
        shuffle(~)
        
        function [outputs, lowestRankComputeLabIndex] = computeInParallel( this, F, NOUT, varargin )
        % computeInParallel  Helper method allowing the user to define and
        % execute a compute function on the parallel pool that contains a
        % dispatch loop implemented much as if it were being run on the
        % client. The arguments to function F must be passed this
        % DistributedDispatcher object, or its DistributedData property,
        % but can otherwise be completely general. Returns a Composite,
        % with each element containing cell array of the outputs of the
        % user function on each worker. If F has reduced the result to
        % labindex 1 then second output lowestRankComputeIndex identifies
        % that worker in the Composite.
        
            data = this.DistributedData;
            % Find the data in the list of arguments
            dataArgs = cellfun(@(arg)isequal(arg,this)||isequal(arg,data), varargin);
            % Delete the data - we can't pass Composites into the SPMD
            % block inside a container
            varargin{dataArgs} = [];
            cellfun(@class, varargin, 'UniformOutput', false);
            spmd
                % Put the data back into the input arguments
                varargin{dataArgs} = data;
                [ outputs{1:NOUT} ] = data.compute(F, NOUT, varargin{:});
                lowestRankComputeLabIndex = data.LowestRankComputeLabIndex;
                lowestRankComputeLabIndex = distributedutil.AutoTransfer( lowestRankComputeLabIndex, lowestRankComputeLabIndex );
            end
            lowestRankComputeLabIndex = lowestRankComputeLabIndex.Value;
        end

    end
    
end

function proportions = iCalculateProportions( workerLoad )
totalWork = sum( workerLoad );
assert( totalWork > 0 );
proportions = workerLoad / totalWork;
end
