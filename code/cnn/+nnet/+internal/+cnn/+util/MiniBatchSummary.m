classdef MiniBatchSummary < nnet.internal.cnn.util.Summary
    % MiniBatchSummary   Class to hold a mini batch training summary
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties
        % Epoch (int)   Number of current epoch
        Epoch
        
        % Iteration (int)   Number of current iteration
        Iteration
        
        % Time (double)   Time spent since training started
        Time
        
        % Loss (double)   Current loss
        Loss
        
        % ValidationLoss (double)   Current validation loss
        ValidationLoss
        
        % LearnRate (double)   Current learning rate
        LearnRate
    end
    
    properties (Dependent)
        % Predictions   4-D array of network predictions
        Predictions
        
        % Response   4-D array of responses
        Response
        
        % ValidationPredictions   4-D array of validation predictions
        ValidationPredictions
        
        % ValidationResponse   4-D array of validation responses
        ValidationResponse
    end
    
    properties (Dependent, SetAccess = protected)  
        % Accuracy (double)   Current accuracy for a classification problem
        Accuracy = [];
        
        % RMSE (double)   Current RMSE for a regression problem
        RMSE = [];
        
        % ValidationAccuracy (double)   Current validation accuracy for a
        % classification problem
        ValidationAccuracy = [];
        
        % ValidationRMSE (double)   Current validation RMSE for a
        % regression problem
        ValidationRMSE = [];
    end
    
    properties (Access = private)
        PrivateAccuracy
        PrivateRMSE
        PrivateValidationAccuracy
        PrivateValidationRMSE
        PrivatePredictions
        PrivateResponse
        PrivateValidationPredictions
        PrivateValidationResponse
        PerformanceStrategy
        MaxIterations
    end
    
    methods
        function this = MiniBatchSummary(varargin)
            this.PerformanceStrategy = nnet.internal.cnn.util.ImagePerformanceStrategy();
            if nargin >= 1
                dispatcher = varargin{1};
                if isa(dispatcher, 'nnet.internal.cnn.sequence.SequenceDispatcher')
                    this.PerformanceStrategy = nnet.internal.cnn.util.VectorPerformanceStrategy();
                end
                if nargin >= 2
                    maxEpochs = varargin{2};
                    this.setMaxIterations(dispatcher, maxEpochs);
                end
            end
        end
        function update( this, predictions, response, epoch, iteration, elapsedTime, miniBatchLoss, learnRate )
            % update   Use this function to update all the
            % properties of the class without having to individually fill
            % in each property.
            this.Predictions = predictions;
            this.Response = response;
            this.Epoch = epoch;
            this.Iteration = iteration;
            this.Time = elapsedTime;
            this.Loss = miniBatchLoss;
            this.LearnRate = learnRate;
        end
        
        function set.Loss( this, loss )
            this.Loss = gather(loss);
        end
        
        function set.ValidationLoss( this, loss )
            this.ValidationLoss = gather(loss);
        end
        
        function accuracy = get.Accuracy( this )
            % get.Accuracy   Get the current accuracy. If the accuracy is
            % empty, recompute it using Predictions and Response.
            this.updateField( 'PrivateAccuracy', ...
                'accuracy', this.Predictions, this.Response );
            
            accuracy = this.PrivateAccuracy;
        end
        
        function set.Accuracy( this, accuracy )
            this.PrivateAccuracy = gather(accuracy);
        end
        
        function accuracy = get.ValidationAccuracy( this )
            % get.ValidationAccuracy   Get the current validation accuracy.
            % If the accuracy is empty, recompute it using
            % ValidationPredictions and ValidationResponse.
            this.updateField( 'PrivateValidationAccuracy', ...
                'accuracy', this.ValidationPredictions, this.ValidationResponse );
            
            accuracy = this.PrivateValidationAccuracy;
        end
        
        function set.ValidationAccuracy( this, accuracy )
            this.PrivateValidationAccuracy = gather(accuracy);
        end
        
        function rmse = get.RMSE( this )
            % get.RMSE   Get the current RMSE. If the RMSE is empty,
            % recompute it using Predictions and Response.
            this.updateField( 'PrivateRMSE', ...
                'rmse', this.Predictions, this.Response );
            
            rmse = this.PrivateRMSE;
        end
        
        function set.RMSE( this, rmse)
            this.PrivateRMSE = gather(rmse);
        end
        
        function rmse = get.ValidationRMSE( this )
            % get.ValidationRMSE   Get the current validation RMSE. If the
            % RMSE is empty, recompute it using ValidationPredictions and
            % ValidationResponse.
            this.updateField( 'PrivateValidationRMSE', ...
                'rmse', this.ValidationPredictions, this.ValidationResponse );
            
            rmse = this.PrivateValidationRMSE;
        end
        
        function set.ValidationRMSE( this, rmse)
            this.PrivateValidationRMSE = gather(rmse);
        end
        
        function predictions = get.Predictions( this )
            predictions = this.PrivatePredictions;
        end
        
        function set.Predictions( this, predictions )
            % set.Predictions   Set predictions and make sure related
            % metrics go out of sync by setting them to empty
            this.PrivatePredictions = gather(predictions);
            this.PrivateAccuracy = [];
            this.PrivateRMSE = [];
        end
        
        function response = get.Response( this )
            response = this.PrivateResponse;
        end
        
        function set.Response( this, response )
            % set.Response   Set response and make sure related metrics go
            % out of sync by setting them to empty
            this.PrivateResponse = gather(response);
            this.PrivateAccuracy = [];
            this.PrivateRMSE = [];
        end
        
        function predictions = get.ValidationPredictions( this )
            predictions = this.PrivateValidationPredictions;
        end
        
        function set.ValidationPredictions( this, predictions )
            % set.ValidationPredictions   Set validation predictions and
            % make sure related metrics go out of sync by setting them to
            % empty
            this.PrivateValidationPredictions = gather(predictions);
            this.PrivateValidationAccuracy = [];
            this.PrivateValidationRMSE = [];
        end
        
        function response = get.ValidationResponse( this )
            response = this.PrivateValidationResponse;
        end
        
        function set.ValidationResponse( this, response )
            % set.ValidationResponse   Set validation response and make
            % sure related metrics go out of sync by setting them to empty
            this.PrivateValidationResponse = gather(response);
            this.PrivateValidationAccuracy = [];
            this.PrivateValidationRMSE = [];
        end
        
        function gather( this )
            % gather  Ensure all properties are stored on the host
            this.PrivatePredictions = gather(this.PrivatePredictions);
            this.PrivateResponse = gather(this.PrivateResponse);
            this.Epoch = gather(this.Epoch);
            this.Iteration = gather(this.Iteration);
            this.Time = gather(this.Time);
            this.Loss = gather(this.Loss);
            this.ValidationLoss = gather(this.ValidationLoss);
            this.LearnRate = gather(this.LearnRate);
            this.PrivateAccuracy = gather(this.PrivateAccuracy);
            this.PrivateRMSE = gather(this.PrivateRMSE);
            this.PrivateValidationAccuracy = gather(this.PrivateValidationAccuracy);
            this.PrivateValidationPredictions = gather(this.PrivateValidationPredictions);
            this.PrivateValidationResponse = gather(this.PrivateValidationResponse);
            this.PrivateValidationRMSE = gather(this.PrivateValidationRMSE);
        end
        
        function tf = isLastIteration(this)
            tf = ~isempty(this.MaxIterations) && this.Iteration == this.MaxIterations;
        end
            
    end
    
    methods (Access = private)
        function updateField( this, field, fcn, predictions, response )
            % updateField   Update field value according to a function fcn
            % of predictions and response. If any of those are empty, the
            % field value will be empty. If the field value is not empty,
            % do not update it
            val = this.(field);
            if isempty( val )
                if isempty(predictions) || isempty(response)
                    val = [];
                else
                    val = gather( this.PerformanceStrategy.(fcn)(predictions, response) );
                end
            else
                % Value is not empty, do not update
            end
            this.(field) = val;
        end
        
        function setMaxIterations(this, data, maxEpochs)
            % setMaxIterations  Provide information to the summary that
            % allows it to know when we are on the last iteration. If this
            % is defined it can be used by reporters to do something
            % different on the last iteration.
            assert(isa(data, 'nnet.internal.cnn.DataDispatcher'));
            miniBatchSize = data.MiniBatchSize;
            numObservations = data.NumObservations;
            endOfEpoch = data.EndOfEpoch;
            iterationsPerEpoch = numObservations / miniBatchSize;
            if endOfEpoch == "truncateLast"
                this.MaxIterations = maxEpochs * ceil(iterationsPerEpoch);
            else
                this.MaxIterations = maxEpochs * floor(iterationsPerEpoch);
            end
        end
    end
end