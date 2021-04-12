classdef (Abstract) SequenceDispatcher < nnet.internal.cnn.DataDispatcher
    % SequenceDispatcher   Interface for sequence data dispatchers
    
    %   Copyright 2017 The MathWorks, Inc.
    
     properties (Abstract, SetAccess = private)
        % SequenceLength   Strategy to determine the length of the
        % sequences used per mini-batch. Options are:
        %       - 'longest' to pad all sequences in a batch to the length
        %       of the longest sequence
        %       - 'shortest' to truncate all sequences in a batch to the
        %       length of the shortest sequence
        %       -  Positive integer - Pad sequences to the have same length
        %       as the longest sequence, then split into smaller sequences
        %       of the specified length. If splitting occurs, then the
        %       function creates extra mini-batches
        SequenceLength
        
        % DispatcherFormat (char)   Format of the response. Either:
        %       'seq2one'  : classify the entire sequence. Response is
        %       numObs-by-1 categorical array.
        %       'seq2seq'  : classify each time step. Response size
        %       numObs-by-1 cell array, containing 1-by-SequenceLength
        %       categorical arrays.        
        %       'predict'  : response is empty
        DispatcherFormat
        
        % NextStrategy (nnet.internal.cnn.sequence.NextStrategy)   Strategy
        % class which determines how mini-batches are prepared, based on:
        %       - DispatcherFormat
        %       - SequenceLength
        %       - readStrategy
        NextStrategy
    end
    
    properties (Abstract, Access = ?nnet.internal.cnn.DistributableDispatcher)
        % Data  (cell array)     A copy of the data in the workspace. This
        % is a numObservations-by-1 cell array, which for each observation
        % contains a DataSize-by-sequenceLength numeric array.
        Data
        
        % Response   A copy of the response data in the workspace. Either:
        % - numObservations-by-1 categorical vector 
        % - numObservations-by-1 cell array, which for each observation
        % contains a 1-by-sequenceLength categorical vector.
        Response
    end
    
    properties (SetAccess = private)
        % ImageSize  Not used by sequence dispatchers
        ImageSize
        
        % IsDone (logical)   True if there is no more data to dispatch
        IsDone
        
        % IsNextMiniBatchSameObs (logical)   False if the next mini-batch
        %                           corresponds to a new set of
        %                           observations. True if, for a given set
        %                           of observations, more data is to be
        %                           dispatched in the sequence dimension
        %                           before moving on to the next set of
        %                           observations.
        IsNextMiniBatchSameObs
    end
    
    properties (Access = protected)
        % StartIndexOfCurrentMiniBatch (int) Start index of current mini
        % batch
        StartIndexOfCurrentMiniBatch
        
        % EndIndexOfCurrentMiniBatch (int) End index of current mini batch
        EndIndexOfCurrentMiniBatch
        
        % OrderedIndices   Order to follow when indexing into the data.
        % This can keep a shuffled version of the indices.
        OrderedIndices
        
        % StartIndexOfCurrentSequence (int) Start index of the current
        % sequence
        StartIndexOfCurrentSequence
        
        % EndIndexOfCurrentSequence (int) End index of the current sequence
        EndIndexOfCurrentSequence
        
        % PrivateMiniBatchSize (int)   Number of elements in a mini batch
        PrivateMiniBatchSize
    end
    
    properties (Dependent)
        % MiniBatchSize (int)   Number of elements in a mini batch
        MiniBatchSize
    end
    
    methods
        function [miniBatchData, miniBatchResponse, miniBatchIndices] = next(this)
            % next   Get the data and response for the next mini batch and
            % correspondent indices
            
            % Map the indices into data and sequence dimension
            [miniBatchIndices, sequenceIndices] = this.computeIndices();
            
            % Read the data
            [miniBatchData, miniBatchResponse, advanceSequence] = this.NextStrategy.nextBatch( ...
                this.Data, this.Response, miniBatchIndices, sequenceIndices);
            
            % Cast the data
            miniBatchData = this.Precision.cast( miniBatchData );
            % Cast the response
            miniBatchResponse = this.Precision.cast( miniBatchResponse );
            
            if any(advanceSequence)
                this.advanceCurrentSequenceIndices();
                this.IsNextMiniBatchSameObs = true;
            else
                this.resetCurrentSequenceIndices();
                this.IsNextMiniBatchSameObs = false;
                % Advance indices of current mini batch
                this.advanceCurrentMiniBatchIndices();
            end
        end
        
        function start(this)
            % start   Go to first mini batch
            this.IsDone = false;
            this.IsNextMiniBatchSameObs = false;
            this.StartIndexOfCurrentMiniBatch = 1;
            this.EndIndexOfCurrentMiniBatch = this.MiniBatchSize;
            
            this.resetCurrentSequenceIndices();
        end
        
        function shuffle(this)
            % shuffle   Shuffle the data
            this.OrderedIndices = randperm(this.NumObservations);
        end
        
        function value = get.MiniBatchSize(this)
            value = this.PrivateMiniBatchSize;
        end
        
        function set.MiniBatchSize(this, value)
            value = min(value, this.NumObservations);
            this.PrivateMiniBatchSize = value;
        end
    end
    
    methods (Access = protected)
        function advanceCurrentMiniBatchIndices(this)
            % advanceCurrentMiniBatchIndices   Move forward start and end
            % index of current mini batch based on mini batch size
            if this.EndIndexOfCurrentMiniBatch == this.NumObservations
                % We are at the end of a cycle
                this.IsDone = true;
            elseif this.EndIndexOfCurrentMiniBatch + this.MiniBatchSize > this.NumObservations
                % Last mini batch is smaller
                if isequal(this.EndOfEpoch, 'truncateLast')
                    this.StartIndexOfCurrentMiniBatch = this.StartIndexOfCurrentMiniBatch + this.MiniBatchSize;
                    this.EndIndexOfCurrentMiniBatch = this.NumObservations;
                else % discardLast
                    % Move the starting index after the end, so that the
                    % dispatcher will return empty data
                    this.StartIndexOfCurrentMiniBatch = this.EndIndexOfCurrentMiniBatch+1;
                    this.IsDone = true;
                end
            else
                % We are in the middle of a cycle
                this.StartIndexOfCurrentMiniBatch = this.StartIndexOfCurrentMiniBatch + this.MiniBatchSize;
                this.EndIndexOfCurrentMiniBatch = this.EndIndexOfCurrentMiniBatch + this.MiniBatchSize;
            end
        end
        
        function advanceCurrentSequenceIndices(this)
            % advanceCurrentSequenceIndices   Move forward start and end
            % index in the sequence dimension
            this.StartIndexOfCurrentSequence = this.StartIndexOfCurrentSequence + this.SequenceLength;
            this.EndIndexOfCurrentSequence = this.EndIndexOfCurrentSequence + this.SequenceLength;
        end
        
        function resetCurrentSequenceIndices(this)
            % resetCurrentSequenceIndices   Take sequence indices back to
            % their initial values
            this.StartIndexOfCurrentSequence = 1;
            if isnumeric(this.SequenceLength)
                this.EndIndexOfCurrentSequence = this.SequenceLength;
            else
                this.EndIndexOfCurrentSequence = [];
            end
        end
        
        function [dataIndices, seqIndices] = computeIndices(this)
            % computeIndices    Compute the indices into the data from
            % start and end index, and compute sequence indices
            
            dataIndices = this.StartIndexOfCurrentMiniBatch:this.EndIndexOfCurrentMiniBatch;
            
            % Convert sequential indices to ordered (possibly shuffled) indices
            dataIndices = this.OrderedIndices(dataIndices);
            
            % Compute sequence indices
            seqIndices = this.StartIndexOfCurrentSequence:this.EndIndexOfCurrentSequence;
        end
    end
end