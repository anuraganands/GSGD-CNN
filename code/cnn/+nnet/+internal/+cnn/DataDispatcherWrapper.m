classdef( Abstract, Hidden ) DataDispatcherWrapper < nnet.internal.cnn.DataDispatcher
% DataDispatcherWrapper   Simple wrapper that calls through to the
% underlying dispatcher

%   Copyright 2017 The MathWorks, Inc.
    
    properties (SetAccess = private)
        
        % ImageSize  (1x3 int) Size of each image to be dispatched
        ImageSize
        
        % ResponseSize   (1x3 int) Size of each response to be dispatched
        ResponseSize
        
        % NumObservations   (int) Number of observations in the data set
        NumObservations
        
        % ClassNames (cellstr) Array of class names corresponding to
        %            training data labels. This is empty for a regression
        %            problem.
        ClassNames
        
        % ResponseNames (cellstr) Array of response names corresponding to
        %               training data response names.
        ResponseNames
    end
    
    properties
        % Precision Precision used for dispatched data
        Precision
        
        % EndOfEpoch    End of epoch strategy
        %
        % Strategy for how to cope with the last mini-batch when the number
        % of observations is not divisible by the number of mini batches.
        %
        % Allowed values: 'truncateLast', 'discardLast'
        EndOfEpoch
        
        % MiniBatchSize   (int) Number of elements in a mini batch.
        MiniBatchSize
    end
    
    properties (Access = protected)
        % Dispatcher  The underlying dispatcher being wrapped
        Dispatcher
    end

    methods
        
        function this = DataDispatcherWrapper( dispatcher )
        % RemoteDispatcher  Wrap input dispatcher
            
            % Copy the dispatcher's properties so that this object appears
            % the same to any users
            this.Dispatcher = dispatcher;
            this.NumObservations = dispatcher.NumObservations;
            this.ClassNames = dispatcher.ClassNames;
            this.ResponseNames = dispatcher.ResponseNames;
            this.MiniBatchSize = dispatcher.MiniBatchSize;
            this.EndOfEpoch = dispatcher.EndOfEpoch;
            this.Precision = dispatcher.Precision;
            this.ImageSize = dispatcher.ImageSize;
            this.ResponseSize = dispatcher.ResponseSize;

        end

    end
    
end