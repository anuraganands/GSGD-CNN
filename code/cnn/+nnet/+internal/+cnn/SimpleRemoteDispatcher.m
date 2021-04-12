classdef( Hidden ) SimpleRemoteDispatcher < ...
        nnet.internal.cnn.DataDispatcherWrapper & ...
        nnet.internal.cnn.RemoteDispatchAdapter
% SimpleRemoteDispatcher   Simple wrapper that calls through to the
% underlying dispatcher, with some collective operations

%   Copyright 2017 The MathWorks, Inc.
    
    properties (SetAccess = private)
        IsDone = false
    end
    
    % Properties for determining loop length - allows dispatchers to avoid
    % needing to communicate to know when the data is finished
    properties (Access = private)
        MaxNumMiniBatches
        NumMiniBatches
        CurrentBatch = 0
    end
    
    methods
        
        function this = SimpleRemoteDispatcher( dispatcher, numObservations, miniBatchSize, endOfEpoch )
        % SimpleRemoteDispatcher  Constructor, needs to be provided with
        % some information to determine loop length because this cannot be
        % gleaned from the underlying dispatcher which represents only a
        % portion of the dataset.
            this = this@nnet.internal.cnn.DataDispatcherWrapper( dispatcher );
            this.IsComputeWorker = this.NumObservations > 0;

            % Ensure the underlying dispatcher always keeps retrieving data
            % until it has run out - this wrapper takes over the job of
            % determining when the iterator has reached the end of the data
            this.Dispatcher.EndOfEpoch = 'truncateLast';
            
            % Determine the number of mini-batches in total
            this.MaxNumMiniBatches = numObservations / miniBatchSize;
            
            % True end of epoch overrides the underlying dispatcher value.
            % This will also set the NumMiniBatches property.
            this.setEndOfEpoch(endOfEpoch);
        end
        
        % next  Retrieve next batch of data
        function [miniBatchData, miniBatchResponse, miniBatchIndices] = next( this )
            if this.CurrentBatch < this.NumMiniBatches
                this.CurrentBatch = this.CurrentBatch + 1;
                [miniBatchData, miniBatchResponse, miniBatchIndices] = this.Dispatcher.next();
            else
                [miniBatchData, miniBatchResponse, miniBatchIndices] = deal([], [], []);
            end
            if this.CurrentBatch >= this.NumMiniBatches
                this.IsDone = true;
            end
        end
        
        % start  Set the next the batch to be the first batch in the epoch
        function start(this)
            this.CurrentBatch = 0;
            this.IsDone = false;
            this.Dispatcher.start();
        end
        
        % shuffle  Shuffle the data
        function shuffle(this)
            this.Dispatcher.shuffle();
        end
        
        % setEndOfEpoch  Set the NumBatches property based on the
        % EndOfEpoch settings
        function setEndOfEpoch(this, endOfEpoch)
            this.EndOfEpoch = endOfEpoch;
            switch(endOfEpoch)
                case 'truncateLast'
                    this.NumMiniBatches = ceil( this.MaxNumMiniBatches );
                case 'discardLast'
                    this.NumMiniBatches = floor( this.MaxNumMiniBatches );
                otherwise
                    assert(false, sprintf('EndOfEpoch type ''%s'' is not supported', endOfEpoch));
            end
        end

    end
    
end