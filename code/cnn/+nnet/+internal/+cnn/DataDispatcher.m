classdef (Abstract) DataDispatcher < handle
    % DataDispatcher   Interface for data dispatching iterators
    %
    % Usage example:
    %   dispatcher.start();
    %   dispatcher.shuffle();
    %   while ~dispatcher.IsDone
    %       [x, t, i] = dispatcher.next();
    %       y(i) = predict( net, x );
    %       accuracy(t, y(i))
    %   end
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    properties (Abstract, SetAccess = private)
        
        % ImageSize  (1x3 int) Size of each image to be dispatched
        ImageSize
        
        % ResponseSize   (1x3 int) Size of each response to be dispatched
        ResponseSize
        
        % NumObservations   (int) Number of observations in the data set
        NumObservations
        
        % IsDone    (logical) True if there is no more data to dispatch
        IsDone
        
        % ClassNames (cellstr) Array of class names corresponding to
        %            training data labels. This is empty for a regression
        %            problem.
        ClassNames
        
        % ResponseNames (cellstr) Array of response names corresponding to
        %               training data response names.
        ResponseNames        
    end
    
    properties (Abstract)
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
    
    methods (Abstract)
        % next  Get data and response relative to the next mini batch and
        % corresponding indices. This follows the iterator pattern and
        % advances the indices of the dispatcher to the next mini batch
        %
        % Syntax:
        %   [inputs, response, indices] = dispatcher.next();
        %
        % Outputs:
        % inputs   - array of double of size [H, W, C, N]
        %            Where C is the number of channels for the input image,
        %            N is the number of observations in the next mini
        %            batch.
        % response - array of double of size [HR, WR, CR, N].
        %            In the case of a classification problem, each response
        %            is dummified, i.e., response(i,j)=1 if observation i
        %            is in class j, and zero otherwise. In that case, HR=1,
        %            WR=1 and CR will be the number of classes in the
        %            dataset.
        % indices  - array of int of size [N, 1]
        
        [miniBatchData, miniBatchResponse, miniBatchIndices] = next(this)
        
        % start     Set the next the mini batch to be the first mini batch
        % in the epoch
        start(this)
        
        % shuffle   Shuffle the data
        shuffle(this)
        
    end
    
end