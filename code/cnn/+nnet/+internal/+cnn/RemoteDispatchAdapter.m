classdef( Abstract, Hidden ) RemoteDispatchAdapter < handle
% RemoteDispatchAdapter  Mixin providing an API to make it easy to use the
% distributed dispatcher types inside a dispatch loop

%   Copyright 2017 The MathWorks, Inc.

    properties( SetAccess = protected )
        % IsComputeWorker  Allows an implementation to specify which
        % workers are front-end dispatchers to include in dispatch loop
        IsComputeWorker

        % ComputeCommunicator  An MPI communicator object for communicating
        % between all compute workers
        ComputeCommunicator
        
        % LowestRankComputeLabIndex  Records which compute worker (in the
        % global communicator) has the lowest index, for retrieving results
        LowestRankComputeLabIndex
    end
        
    properties( Access = protected )
        % IsInitialized  Ensures initialize/finalize always match across
        % the pool even in the presence of errors
        IsInitialized = false
    end
    
    methods
        
        function initialize( this )
        % initialize  Call before dispatching any data. The default
        % implementation creates a communicator to isolate the compute
        % workers which are (from a user's point of view) dispatching the
        % data.
            labIndexWithData = labindex;
            if ~this.IsComputeWorker
                labIndexWithData = inf;
            end
            this.LowestRankComputeLabIndex = gop(@min, labIndexWithData);
            this.ComputeCommunicator = distributedutil.CommSplitter( ...
                double(this.IsComputeWorker)+1, labindex );
            % At this point pool is synchronized because the preceding
            % operations are collective. If the last operation was
            % successful it's essential that finalize() is called to
            % synchronize the pool before exiting, or we can have
            % mismatched communication
            this.IsInitialized = true;
        end
        
        function finalize( this )
        % finalize   Call after the dispatch loop. The default
        % implementation only needs to revert to the original communicator.
            if this.IsInitialized
                delete(this.ComputeCommunicator);
            end
            this.IsInitialized = false;
        end
        
        function varargout = compute( this, F, NOUT, varargin )
        % compute  Makes the job of using a remote dispatcher easier, by
        % wrapping a function in the initialize/finalize bookends, and
        % guarding against dispatch from non-compute workers
            initialize( this );
            varargout = cell(1,NOUT);
            if this.IsComputeWorker
                [ varargout{1:NOUT} ] = F( varargin{:} );
            end
            finalize( this )
        end
        
    end
    
    methods( Abstract )
        
        % setEndOfEpoch  Force users of this interface to define a way to
        % change the loop length via the endOfEpoch setting
        setEndOfEpoch( this, endOfEpoch )
        
    end

end
