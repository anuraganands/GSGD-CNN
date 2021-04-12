classdef( Hidden, Sealed ) ParallelBackgroundPrefetcher < ...
        nnet.internal.cnn.RemoteDispatchAdapter & ...
        nnet.internal.cnn.ParallelBackgroundAdapter
% ParallelBackgroundPrefetcher  The back-end for ParallelBackground
% dispatch, prefetching data from the underlying dispatchers and sending
% them to the compute workers

%   Copyright 2017 The MathWorks, Inc.
    
    properties (Constant)
        % MaxBufferedBatches   How many batches in advance should the
        % background workers buffer? Because dispatch is synchronous there
        % can be no advantage to buffering in terms of ensuring background
        % workers are always busy, but it can help prevent delays caused by
        % variability in how long each batch takes to fetch. Set this too
        % high and you impose startup and storage overhead.
        MaxBufferedBatches = 2
    end
    
    properties( Access = private )
        % Dispatcher  The dispatchers for each compute worker
        Dispatcher
        
        % StopFetching  Signal to exit the prefetch loop
        StopFetching = false
        
        % DataBuffer  Stores data for the next and future minibatches
        DataBuffer
    end
    
    methods
        
        function this = ParallelBackgroundPrefetcher( dispatcher, varargin )
        % ParallelBackgroundPrefetcher  Constructor
        
            this.Dispatcher = dispatcher;
            this.IsComputeWorker = false;

            % Ensure the underlying dispatchers always keep retrieving data
            % until it has run out - this wrapper takes over the job of
            % determining when the iterator has reached the end of the data
            for d = 1:numel(this.Dispatcher)
                this.Dispatcher(d).EndOfEpoch = 'truncateLast';
            end
        end
                
        function initialize( this )
        % initialize  Modifies the base implementation to initiate a spin
        % cycle on this worker, starting off continuous prefetch
            this.debugPrint('Lab %d called initialize\n', labindex);

            this.StopFetching = false;
            initialize@nnet.internal.cnn.RemoteDispatchAdapter( this );
            if ~isempty(this.LabMapping)
                this.debugPrint('Lab %d calling spin\n', labindex);
                spin( this );
            end
        end
        
        function finalize( this )
        % finalize  Modifies the base implementation to ensure the
        % background and compute workers are synchronized
            this.debugPrint('Lab %d entering finalize\n', labindex);
            
            flush( this );
            this.debugPrint('Lab %d finished flush\n', labindex);
            finalize@nnet.internal.cnn.RemoteDispatchAdapter( this );
            this.debugPrint('Lab %d exiting finalize\n', labindex);
        end
        
    end
    
    methods (Access = private)

        function next( this )
        % next   Send the next sub-batch in the schedule, as well as
        % populating the buffer with data prefetched from future batches

            % Switch to the inter-node communicator
            commLifeObj = this.NodeCommunicator.select();
            this.debugPrint('Lab %d in next\n', labindex);
            
            % If no compute worker is waiting for data, fill the
            % buffer, otherwise send it immediately if there is
            % buffered data
            partner = this.ComputeLabs(this.LabMapping(this.NextSubBatch,2));
            bufferSize = numel( unique( [ this.DataBuffer.MiniBatchIndex ] ) );
            while bufferSize < this.MaxBufferedBatches && ...
                    (bufferSize == 0 || ~labProbe( partner, this.CommsTag ))
                % Loop over all the sub-batches being served by this
                % background worker (happens when there are fewer
                % background workers than compute workers)
                if bufferSize == 0
                    subBatchIndex = this.NextSubBatch;
                else
                    subBatchIndex = this.getNextSubBatchIndex( this.DataBuffer(end).SubBatchIndex );
                end
                currentMiniBatchIndex = this.LabMapping(subBatchIndex, 1);
                miniBatchIndex = currentMiniBatchIndex;
                wrappedAround = false;
                while miniBatchIndex == currentMiniBatchIndex && ~wrappedAround
                    batchData = getSubBatch( this, subBatchIndex );
                    this.DataBuffer = [this.DataBuffer; ...
                        iCreateDataBufferElement( subBatchIndex, miniBatchIndex, batchData ); ];
                    nextBatchIndex = this.getNextSubBatchIndex( subBatchIndex );
                    if nextBatchIndex <= subBatchIndex
                        wrappedAround = true;
                    end
                    miniBatchIndex = this.LabMapping(subBatchIndex, 1);
                    subBatchIndex = nextBatchIndex;
                end
                bufferSize = bufferSize + 1;
            end

            % Serve data to compute worker
            this.debugPrint('Lab %d sending sub-batch %d (mini-batch %d) to lab %d\n', labindex, ...
                this.DataBuffer(1).SubBatchIndex, this.DataBuffer(1).MiniBatchIndex, partner);
            signal = labSendReceive( partner, partner, this.DataBuffer(1).Data, this.CommsTag );
            
            % 'Pop' the sent data off the front of the buffer
            this.DataBuffer(1) = [];
            
            % Respond to signal or simply move to next batch
            this.debugPrint('Lab %d received signal "%s"\n', labindex, signal);
            if signal == "restart"
                restart( this );
            elseif signal == "stop"
                this.StopFetching = true;
            elseif signal == "shuffle"
                shuffle( this );
            else
                % Move the counters on
                this.CurrentSubBatch = this.NextSubBatch;
                this.NextSubBatch = getNextSubBatchIndex(this, this.NextSubBatch);
                this.debugPrint('Lab %d advancing to sub-batch %d\n', labindex, this.NextSubBatch);
            end
            
            % Switch back to original communicator
            delete(commLifeObj);
            
        end
        
        function spin( this )
        % spin   Put this worker into a continuous spin cycle, prefetching
        % and sending data until it receives a stop signal
            this.StopFetching = false;
            this.CurrentSubBatch = 0;
            this.NextSubBatch = 1;
            this.DataBuffer = iCreateDataBufferElement(); % Initialize
            while ~this.StopFetching
                this.debugPrint('Lab %d in spin calling next\n', labindex);
                next( this );
            end
            this.debugPrint('Lab %d exiting spin\n', labindex);
        end
        
        function restart( this )
        % restart  Erase buffered data and reset schedule to start of
        % dataset
            this.debugPrint('Lab %d entering restart\n', labindex);
            this.DataBuffer = iCreateDataBufferElement(); % Erase buffer
            this.NextSubBatch = 1;
        end
        
        function shuffle( this )
        % shuffle  Shuffle all the dispatchers on one of the background
        % workers on each node, and transmit to the others
            
            this.debugPrint('Lab %d entering shuffle\n', labindex);
            this.DataBuffer = iCreateDataBufferElement(); % Erase buffer

            % Switch to the communicator between background workers on this
            % node
            commLifeObj = this.BackgroundNodeCommunicator.select();

            % Shuffle all the dispatchers on one of the background workers
            % and broadcast to the others
            if labindex == 1
                for c = 1:numel(this.ComputeLabs)
                    this.debugPrint('Lab %d is shuffling dispatcher %d\n', this.NodeLabIndex, c);
                    shuffle( this.Dispatcher(c) );
                end
            end
            this.debugPrint('Lab %d finished shuffling\n', this.NodeLabIndex);
            this.Dispatcher = labBroadcast( 1, this.Dispatcher );
            this.debugPrint('Lab %d has new dispatchers\n', this.NodeLabIndex);
            
            % Switch back to original communicator
            delete(commLifeObj);

        end
        
    end
    
    methods( Access = private )
        
        function batchData = getSubBatch( this, subBatchIndex )
        % getSubBatch  Helper that actually gets data from the underlying
        % dispatchers
            this.debugPrint('Lab %d getting subBatch %d\n', labindex, subBatchIndex);
            mapping = this.LabMapping(subBatchIndex, :);
            computeIndex = mapping(2); % Tells us which dispatcher to use
            batchIndex = mapping(1);   % Data is not accessed sequentially
            try
                [ batchData{1}, batchData{2}, batchData{3} ] = ...
                    this.Dispatcher(computeIndex).getBatch(batchIndex);
            catch err
                batchData = err;
                this.debugPrint('Lab %d errored getting subBatch %d\n', labindex, subBatchIndex);
            end
            this.debugPrint('Lab %d finished getting subBatch %d\n', labindex, subBatchIndex);
        end

        function flush( this )
        % flush  Ensure background and compute workers are synchronized
            if this.IsInitialized && ...
                    ~this.StopFetching && ~isempty(this.LabMapping)
                % Background workers may need to call next to help
                % compute workers flush, if they have been initialized
                % but the spin was not stopped by receiving a
                % StopTraining signal. This can happen if they are
                % being interrupted by the client (e.g. Ctrl-C).
                this.debugPrint('Lab %d spinning in order to flush\n', labindex);
                next( this );
            end
        end
        
        function nextBatchIndex = getNextSubBatchIndex( this, currentSubBatchIndex )
        % getNextSubBatchIndex   Helper to compute the next batch index assuming
        % wrap-around
            nextBatchIndex = mod( currentSubBatchIndex, size(this.LabMapping, 1) ) + 1;
            % Wrap around if we're scheduled to retrieve mini-batches that
            % aren't wanted, due to the EndOfEpoch setting
            nextMiniBatchIndex = this.LabMapping(nextBatchIndex, 1);
            if nextMiniBatchIndex > this.NumMiniBatches
                nextBatchIndex = 1;
            end
        end
        
    end
    
end

function buffer = iCreateDataBufferElement(subBatchIndex, miniBatchIndex, data)
% iCreateDataBufferElement  Helper to create DataBuffer struct
if nargin == 0
    subBatchIndex = {};
    miniBatchIndex = {};
    data = {};
end
% Prevent attempt to create struct of arrays.
if iscell(data) && ~isempty(data)
    data = { data };
end
buffer = struct( ...
    'SubBatchIndex', subBatchIndex, ...
    'MiniBatchIndex', miniBatchIndex, ...
    'Data', data );
end