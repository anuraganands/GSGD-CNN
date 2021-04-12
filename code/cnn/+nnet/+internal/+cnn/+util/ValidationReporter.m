classdef (Sealed) ValidationReporter < nnet.internal.cnn.util.Reporter
    % ValidationReporter   Class to hold validation data and compute
    % performance metrics on them
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Access = private)
        % Data (nnet.internal.cnn.DataDispatcher)   A data dispatcher
        Data
        
        % Precision (nnet.internal.cnn.util.Precision)   A precision object
        Precision
        
        % ExecutionSettings   A struct specifying the execution environment
        ExecutionSettings
        
        % Frequency   Frequency to compute validation metrics in iterations
        Frequency
        
        % Patience   Number of times that the loss is allowed to increase
        % or remain unchanged before training is stopped
        Patience
        
        % BestLoss   Smallest loss achieved by the network so far
        BestLoss = Inf;
        
        % StepsWithoutDecrease   Number of steps passed without a decrease
        % of the best loss
        StepsWithoutDecrease = 0;
        
        % ShuffleOption   Controls when to shuffle the data. It can be:
        % 'once', 'every-epoch', or 'never'
        ShuffleOption
    end
    
    methods
        function this = ValidationReporter(data, precision, executionSettings, frequency, patience, shuffleOption)
            this.Data = data;
            this.Precision = precision;
            this.ExecutionSettings = executionSettings;
            this.Frequency = frequency;
            this.Patience = patience;
            this.ShuffleOption = shuffleOption;
            
            if isequal(this.ShuffleOption, 'once')
                this.Data.shuffle();                
            end
        end
        
        function setup( ~ )
        end
        
        function start( ~ )
        end
        
        function reportIteration( this, summary )
            if this.doComputeThisIteration( summary )
                this.updateBestLoss( summary.ValidationLoss );
                this.notifyTrainingInterrupt();
            end
        end
        
        function computeIteration(this, summary, net)
            % computeIteration   Compute predictions on the validation set
            % using the network net according to the current iteration and
            % update the MiniBatchSummary summary
            %
            % summary   - A nnet.internal.cnn.util.MiniBatchSummary object
            % net       - A nnet.internal.cnn.SeriesNetwork object
            
            if this.doComputeThisIteration( summary )
                this.predictAndUpdateSummary( summary, net );
            else
                % Return empty values
                summary.ValidationPredictions = [];
                summary.ValidationResponse = [];
                summary.ValidationLoss = [];
            end

        end
        
        function reportEpoch( ~, ~, ~, ~ )
        end
        
        function finish( ~, ~ )
        end
        
        function computeFinish( this, summary, network )
            % Always compute the validation at the end of training, if it
            % hasn't already been computed
            if ~this.doComputeThisIteration( summary )
                this.predictAndUpdateSummary( summary, network );
            end
        end
        
        function summaryOut = computeFinalValidationResultForPlot(this, summaryIn, net)
            % This method should only ever be called after network has
            % finished training and has been finalized.
            assert(iAmClientMatlab(), 'computeFinalValidationResultForPlot was called on a parallel pool');

            % The summary is a handle object but for safety we copy it
            % (this is necessary for parallel compute anyway)
            if this.ExecutionSettings.useParallel
                computeLabOne = find(this.ExecutionSettings.workerLoad > 0, 1, 'first');
                spmd
                    if labindex == computeLabOne
                        % Don't risk moving things to the GPU until we know
                        % we're on a machine with a GPU
                        net = iSetupNetworkForPredictionInEnvironment(net, this.ExecutionSettings.executionEnvironment);
                        this.predictAndUpdateSummary( summaryIn, net );
                    end
                    % Use feval to avoid issues with name resolution during
                    % SPMD dependency analysis when PCT is not present
                    summaryOut = feval('distributedutil.AutoTransfer', summaryIn, computeLabOne );
                end
                summaryOut = summaryOut.Value;
            else
                net = iSetupNetworkForPredictionInEnvironment(net, this.ExecutionSettings.executionEnvironment);
                this.predictAndUpdateSummary( summaryIn, net );
                summaryOut = summaryIn;
            end
        end
        
    end
    
    methods (Access = private)
        function tf = doComputeThisIteration( this, summary )
            iteration = summary.Iteration;
            tf = mod(iteration, this.Frequency) == 0 || iteration == 1 || summary.isLastIteration();
        end
        
        function predictAndUpdateSummary( this, summary, net )
            [predictions, response] = this.predict( net );
            loss = net.loss( predictions, response );
            summary.ValidationPredictions = predictions;
            summary.ValidationResponse = response;
            summary.ValidationLoss = loss;
        end
        
        function [predictions, response] = predict( this, net )
            predictions = this.allocatePredictionsArray();
            response = this.allocatePredictionsArray();
            % When 'shuffle' is set to 'every-epoch', validation data is
            % shuffled everytime we compute validation metrics. This is
            % because one epoch corresponds to an entire pass over the
            % dataset, so each time we compute a validation iteration using
            % all the data we should shuffle
            if isequal(this.ShuffleOption, 'every-epoch')
                this.Data.shuffle();
            end
            this.Data.start();
            while ~this.Data.IsDone
                [X, Y, idx] = this.Data.next();
                X = this.prepareForExecutionEnvironment(X);
                Y = this.prepareForExecutionEnvironment(Y);
                currentBatchPredictions = net.predict( X );
                % Predictions can be in a cell-array if net is a DAG
                % network
                if iscell( currentBatchPredictions )
                    currentBatchPredictions = currentBatchPredictions{1};
                end
                predictions(:,:,:,idx) = currentBatchPredictions;
                response(:,:,:,idx) = Y;
            end
        end
        
        function X = prepareForExecutionEnvironment( this, X )
            if nnet.internal.cnn.util.GPUShouldBeUsed( this.ExecutionSettings.executionEnvironment )
                % Move data to GPU
                X = gpuArray( X );
            else
                % Do nothing
            end
        end
        
        function predictions = allocatePredictionsArray( this )
            % allocatePredictionsArray   Allocate a prediction array
            % according to the size of the responses and the number of
            % observations
            predictions = this.Precision.zeros([this.Data.ResponseSize this.Data.NumObservations]);
            predictions = this.prepareForExecutionEnvironment( predictions );
        end
        
        function updateBestLoss( this, loss )
            % updateBestLoss   If the loss has decreased, update the best
            % loss. If it hasn't, increase the counter of steps without
            % loss decrease
            
            if loss < this.BestLoss
                this.BestLoss = loss;
                this.StepsWithoutDecrease = 0;
            else
                this.StepsWithoutDecrease = this.StepsWithoutDecrease + 1;
            end
        end
        
        function notifyTrainingInterrupt( this )
            % notifyTrainingInterrupt   Notify listeners of
            % TrainingInterruptEvent if there was no loss decrease for
            % this.Patience steps
            if this.StepsWithoutDecrease >= this.Patience
                notify( this, 'TrainingInterruptEvent' );
            end
        end
    end
end

function tf = iAmClientMatlab()
tf = ~nnet.internal.cnn.util.canUsePCT() || isempty(getCurrentJob());
end

function net = iSetupNetworkForPredictionInEnvironment(net, environment)
net = prepareNetworkForPrediction(net);
if environment == "gpu"
    net = setupNetworkForGPUPrediction(net);
else
    net = setupNetworkForHostPrediction(net);
end
end