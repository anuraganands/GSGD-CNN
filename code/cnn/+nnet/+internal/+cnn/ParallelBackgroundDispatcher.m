classdef( Hidden, Sealed ) ParallelBackgroundDispatcher < ...
        nnet.internal.cnn.DataDispatcherWrapper & ...
        nnet.internal.cnn.RemoteDispatchAdapter & ...
        nnet.internal.cnn.ParallelBackgroundAdapter
% ParallelBackgroundDispatcher  The front-end for ParallelBackground
% dispatch, presenting the DataDispatcher interface to a user but receiving
% data prefetched by the background workers

%   Copyright 2017-2018 The MathWorks, Inc.


    properties (SetAccess = private)
        % IsDone    (logical) True if there is no more data to dispatch
        IsDone = false
    end
    
    properties (Access = private)
        % Signal  Passed to the background workers to trigger events such
        % as stopping the dispatch loop
        Signal = "run"
    end
    
    methods
        
        function this = ParallelBackgroundDispatcher( dispatcher )
        % ParallelBackgroundDispatcher  Constructor. Copies the properties
        % of the underlying dispatcher but discards it.
            
            % Check that this environment and dispatcher supports
            % background dispatch
            nnet.internal.cnn.BackgroundDispatcher.checkCanRunInBackground( dispatcher );

            % Copy properties via the base class
            this = this@nnet.internal.cnn.DataDispatcherWrapper( dispatcher );
            % Now the internal properties are set we can delete the
            % dispatcher
            this.Dispatcher = [];

            this.IsComputeWorker = true;
        end

        function [miniBatchData, miniBatchResponse, miniBatchIndices] = next( this )
        % next   Get the data and response for the next mini batch

            this.debugPrint('Lab %d in next\n', labindex);
            [miniBatchData, miniBatchResponse, miniBatchIndices] = receive( this, this.NextSubBatch );
            
            % Move the counters on
            if ~this.IsDone && this.NextSubBatch == this.NumMiniBatches
                % Latch IsDone to true until cleared by start()
                this.IsDone = true;
            end
            this.CurrentSubBatch = this.NextSubBatch;
            this.NextSubBatch = getNextSubBatchIndex(this, this.NextSubBatch);
            this.debugPrint('Lab %d advancing to sub-batch %d\n', labindex, this.NextSubBatch);
            
        end
        
        function start( this )
        % start  Move read point to start

            this.debugPrint('Lab %d entering start\n', labindex);

            % Make sure we're not flushing unnecessarily at the start of
            % each loop
            if this.CurrentSubBatch ~= 0 && this.NextSubBatch ~= 1
                this.Signal = "restart";
                flush( this );
                this.NextSubBatch = 1;
            end
            this.IsDone = false;
            
        end
        
        function shuffle( this )
        % shuffle  Signal the background workers to shuffle the data
            
            this.debugPrint('Lab %d entering shuffle\n', labindex);

            % Sync compute with background
            this.Signal = "shuffle";
            flush( this );
        end
        
        function finalize( this )
        % finalize  Overload to ensure the pool is synced and the
        % background prefetch spin cycle has been terminated
            this.debugPrint('Lab %d entering finalize\n', labindex);
            
            this.Signal = "stop";
            flush( this );
            this.debugPrint('Lab %d finished flush\n', labindex);
            finalize@nnet.internal.cnn.RemoteDispatchAdapter( this );
            this.debugPrint('Lab %d exiting finalize\n', labindex);
        end
        
        function setEndOfEpoch( this, endOfEpoch )
        % setEndOfEpoch  Overload to set property as well as call base
        % class implementation to change the length of the dispatch loop
            setEndOfEpoch@nnet.internal.cnn.ParallelBackgroundAdapter( this, endOfEpoch );
            this.EndOfEpoch = endOfEpoch;
        end
        
    end 
    
    methods( Access = private )
        

        function [miniBatchData, miniBatchResponse, miniBatchIndices] = receive( this, miniBatchIndex )
        % receiveNext   Get the data and response for a given mini batch

            % Switch to the inter-node communicator
            commLifeObj = this.NodeCommunicator.select(); %#ok<NASGU>
            
            % Compute workers read from the appropriate background worker,
            % sending it the current signal to trigger events if needed
            partner = this.LabMapping(miniBatchIndex);
            this.debugPrint('Lab %d receiving batch %d from lab %d\n', labindex, miniBatchIndex, partner);
            batchData = labSendReceive( partner, partner, this.Signal, this.CommsTag );
            if isa( batchData, 'MException' )
                this.debugPrint('Lab %d received error with message %s\n', labindex, batchData.message);
                throw( batchData );
            end
            [miniBatchData, miniBatchResponse, miniBatchIndices] = deal( batchData{:} );
            
        end
            
        function flush( this )
        % flush  Forces the compute workers and background workers to
        % synchronize by resolving any background workers waiting on
        % communication. That is then the opportunity to signal them to
        % change state if necessary.
            assert(~isempty(getCurrentJob()));
            
            % if clause protects against errors during initialization
            if this.IsInitialized
                
                this.debugPrint('Flush Lab %d\n', labindex);

                % Go through the upcoming schedule ensuring every
                % background worker waiting on a sendReceive is serviced.
                % This worker may receive several times or not at all,
                % depending on how many background workers there are and
                % whether or not this flush wraps around to the start of
                % the data.
                numComputeWorkers = numel(this.ComputeLabs);
                numBackgroundWorkers = numel(this.BackgroundLabs);
                waitingBackgroundWorkers = true(1, numBackgroundWorkers);
                miniBatchIndex = this.NextSubBatch;
                while any(waitingBackgroundWorkers)
                    for c = 1:numComputeWorkers
                        computeLab = this.ComputeLabs(c);
                        backgroundLab = this.AllLabsSchedule(miniBatchIndex, c);
                        whichBackgroundLab = this.BackgroundLabs == backgroundLab;
                        if waitingBackgroundWorkers(whichBackgroundLab)
                            if this.NodeLabIndex == computeLab
                                this.debugPrint('Lab %d flushing batch %d from lab %d\n', labindex, miniBatchIndex, backgroundLab);
                                receive(this, miniBatchIndex);
                            end
                            waitingBackgroundWorkers(whichBackgroundLab) = false;
                        end
                    end
                    miniBatchIndex = this.getNextSubBatchIndex(miniBatchIndex);
                end
                
                this.Signal = "run";
            end
        end
        
        function nextBatchIndex = getNextSubBatchIndex( this, currentSubBatchIndex )
        % getNextSubBatchIndex   Helper to compute the next batch index assuming
        % wrap-around
            nextBatchIndex = mod( currentSubBatchIndex, this.NumMiniBatches ) + 1;
        end
        
    end
end
