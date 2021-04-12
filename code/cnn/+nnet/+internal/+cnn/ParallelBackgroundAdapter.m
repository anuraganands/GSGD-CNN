classdef( Abstract, Hidden ) ParallelBackgroundAdapter < handle
% ParallelBackgroundAdapter  Interface for dispatch handling on remote
% workers where some workers are involved in parallel training and others
% are prefetching files.
    
%   Copyright 2017-2018 The MathWorks, Inc.
    
    properties (Constant)
        % CommsTag  Used to ensure communication between background and
        % compute workers is kept distinct
        CommsTag = 101420599
    end
    
    properties( Access = protected )
        
        % CurrentSubBatch  Keeps track of the current sub-batch as we
        % progress through the data. Sub-batch refers to a partition of a
        % mini-batch being processing by one compute worker.
        CurrentSubBatch = 0
        
        % NextSubBatch  Keeps track of the next sub-batch as we progress
        % through the data
        NextSubBatch = 1
        
        % NumMiniBatches  How many mini batches there currently are in a
        % complete dispatch loop; changes when EndOfEpoch setting changes
        NumMiniBatches

        % ComputeLabs  For each node, records which lab indices are compute
        % workers
        ComputeLabs
        
        % BackgroundLabs  For each node, records which lab indices are
        % background workers
        BackgroundLabs
        
        % NodeLabIndex  Index of a worker within the workers of each node
        NodeLabIndex
        
        % AllLabsSchedule  Complete schedule mapping minibatches for every
        % compute worker to a background lab index
        AllLabsSchedule
        
        % LabMapping  For each node, maps every minibatch for each compute
        % worker to a background lab index
        LabMapping
        
        % NodeCommunicator  A split communicator object for communicating
        % between all workers on a node
        NodeCommunicator
        
        % BackgroundNodeCommunicator  A split communicator object for
        % communicating between all background workers on a node
        BackgroundNodeCommunicator
    end
    
    properties( Access = private )
        % MaxNumMiniBatches  How many batches at most can be dispatched in
        % a complete dispatch loop, as a floating point value
        MaxNumMiniBatches
    end
    
    methods ( Static )
        
        function obj = createParallelBackgroundDispatch( computeData, proportions, numObservations, miniBatchSize, endOfEpoch)
        % createParallelBackgroundDispatch  Factory function for background
        % dispatch. Given a set of dispatchers to be dispatched by
        % front-end compute workers, passes them to background workers and
        % sets up a comms schedule so that the compute workers can receive
        % data from them instead. Creates dispatcher wrappers on each
        % worker as front-end dispatchers or background prefetchers as
        % appropriate.
        
            % Note: when referring to 'batch' in the comments it means a
            % sub-batch of data being read from one of the underlying
            % dispatchers and transmitted from a background worker to a
            % compute worker. For compute workers there is one batch per
            % mini-batch, so they can seem interchangeable; but background
            % workers may serve up a sub-batch only every other mini-batch,
            % or they may read multiple sub-batches per mini-batch.
        
            % As well as creating the dispatcher objects, the objective of
            % this function is to create a dispatch schedule, which is
            % separately defined for each node.
            %
            % The objective is to create a lab mapping which identifies,
            % for each batch, the source of the data for each compute
            % worker.
            %
            % Let us say workers [1 2] on this node are compute workers and
            % workers [3 4 5] are background workers. If there are 4
            % mini-batches in this dataset we expect the mapping to be
            %
            %                          computeLabIndex = [ 1   2 ]
            %                                              |   |
            % miniBatchIndex = [ 1     |    labMapping = [ 3   4
            %                    2     |                   5   3
            %                    3     |                   4   5
            %                    4 ]   |                   3   4 ]
            %
            % On the compute workers we strip out the column relevant to
            % this worker. On the background workers we convert to the
            % inverse mapping for each worker and add the mini-batch index.
            % So for worker 5, the resulting mapping will be
            %
            %       miniBatch     computeLabIndex
            %                \   /
            % labMapping = [ 2   1
            %                3   2 ]
            %
            % This tells us that the first batch this worker will dispatch
            % is for mini-batch 2, going to lab 1; the second batch is
            % for mini-batch 3, going to lab 2.
            %
            % Throughout this function I will use this example in comments
            
            % Define the number of mini-batches in the dispatch loop. For
            % the purposes of defining the dispatch schedule we are
            % assuming a 'truncateLast' end-of-epoch policy.
            numMiniBatches = ceil(numObservations ./ miniBatchSize);
            
            % Determine the cluster nodes, compute workers and background
            % workers
            hostid = gcat( {parallel.internal.general.HostNameUtils.getLocalHostAddress()}, 1 );
            [~, ~, ids] = unique(hostid, 'sorted');
            myNode = ids == ids(labindex);
            nWorkersThisNode = sum(myNode);
            computeWorkers = proportions ~= 0;
            isComputeWorker = computeWorkers(labindex);
            computeWorkersThisNode = myNode & computeWorkers(:);
            backgroundWorkersThisNode = myNode & ~computeWorkers(:);
            
            % Create an object implementing the ParallelBackgroundAdapter
            % for each worker. Compute workers get one that just wraps the
            % dispatcher distributed to them. Background workers get one
            % that wraps all the dispatchers for every compute worker on
            % their node.
            if isComputeWorker
                obj = nnet.internal.cnn.ParallelBackgroundDispatcher( ...
                    computeData{labindex} );
            else
                obj = nnet.internal.cnn.ParallelBackgroundPrefetcher( ...
                    cat(1, computeData{computeWorkersThisNode}) );
            end
            
            % Get the (pool) index of all workers on this worker's node,
            % as well as this worker's index within the node
            allLabs = 1:numlabs;
            nodeLabs = allLabs(myNode);
            obj.NodeLabIndex = find(nodeLabs == labindex);
            computeLabs = allLabs(computeWorkersThisNode); % [1 2]

            % Get the list of worker indices within the node for the
            % compute and background workers
            nComputeWorkersThisNode = sum(computeWorkersThisNode);
            nBackgroundWorkersThisNode = nWorkersThisNode - nComputeWorkersThisNode;
            nodeLabs = 1:nWorkersThisNode;
            obj.ComputeLabs = nodeLabs(computeWorkersThisNode(myNode));
            obj.BackgroundLabs = nodeLabs(backgroundWorkersThisNode(myNode));
            numSubBatches = nComputeWorkersThisNode*numMiniBatches;
            % Deal with when there aren't enough batches for every
            % background worker to be needed. This includes stripping
            % workers that will only dispatch when endOfEpoch is
            % 'truncateLast'.
            minNumSubBatches = nComputeWorkersThisNode*(floor(numObservations ./ miniBatchSize));
            if minNumSubBatches < nBackgroundWorkersThisNode
                obj.BackgroundLabs = obj.BackgroundLabs(1:minNumSubBatches);
                nBackgroundWorkersThisNode = minNumSubBatches;
            end
            
            % If this node is disabled there will be no computeWorkers and
            % no sub-batches. Leave the lab mapping empty and the
            % background workers will behave correctly.
            if nComputeWorkersThisNode > 0
                
                % Assign a background worker to every minibatch on every
                % compute worker in a round-robin fashion.
                % Get flat list first e.g. [ 3 4 5 3 4 5 3 4 5 ]';
                labMapping = repmat( obj.BackgroundLabs(:), ceil(numSubBatches/nBackgroundWorkersThisNode) );
                % Crop superfluous e.g. [ 3 4 5 3 4 5 3 4 ]'
                labMapping = labMapping(1:numSubBatches);
                % Reshape giving one row per compute worker e.g. [3 5 4 3
                %                                                 4 3 5 4]
                labMapping = reshape( labMapping, nComputeWorkersThisNode, numMiniBatches );
                % Get compute worker index for each sub-batch
                computeMapping = repmat( (1:nComputeWorkersThisNode)', 1, numMiniBatches );
                % Get mini-batch index for each sub-batch
                batchMapping = repmat( 1:numMiniBatches, nComputeWorkersThisNode, 1 );
                
                % Adjust the mapping into the form needed for each worker type
                obj.AllLabsSchedule = labMapping';
                if isComputeWorker
                    obj.LabMapping = labMapping(computeLabs == labindex, :)';
                else
                    % Calculate inverse mapping, ie. for each background
                    % worker get the compute worker for each subbatch
                    mappingTable = [ batchMapping(:), computeMapping(:) ];
                    obj.LabMapping = mappingTable(labMapping == obj.NodeLabIndex, :);
                end
                
            end
            
            % Set the loop variables which define how many batches are
            % dispatched in each loop
            obj.MaxNumMiniBatches = numObservations ./ miniBatchSize;
            setEndOfEpoch(obj, endOfEpoch);
            
            % Finally, create communicators that will allow workers on
            % each node to communicate. The communicators are not selected
            % here, just created.
            obj.NodeCommunicator = nnet.internal.cnn.util.Communicator( ids(labindex) );
            nodeIdBackgroundOnly = ids(labindex);
            if isComputeWorker || isempty(obj.LabMapping)
                % Push all other workers into a 'reject' group
                nodeIdBackgroundOnly = max(ids) + 1;
            end
            obj.BackgroundNodeCommunicator = nnet.internal.cnn.util.Communicator( nodeIdBackgroundOnly );
        end
        
    end
    
    methods
        
        function setEndOfEpoch(this, endOfEpoch)
        % setEndOfEpoch  Change the length of the dispatch loop based
        % on the EndOfEpoch setting
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
    
    methods( Access = protected )
        function debugPrint( ~, varargin )
            if nnet.internal.cnn.ParallelBackgroundAdapter.doDebug()
                fprintf( varargin{:} );
            end
        end
    end
    
    methods( Static )
        function tf = doDebug( tf )
            persistent DebugEnabled
            if isempty(DebugEnabled)
                DebugEnabled = false;
            end
            if nargin > 0
                DebugEnabled = tf;
            end
            tf = DebugEnabled;
        end
    end
end
