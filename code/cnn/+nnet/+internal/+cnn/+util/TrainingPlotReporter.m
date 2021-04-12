classdef TrainingPlotReporter < nnet.internal.cnn.util.Reporter
    % TrainingPlotReporter   Reporter for the training plot.
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties(Access = private)
        % ValidationReporter   (nnet.internal.cnn.util.ValidationReporter)
        % ValidationReporter used to compute the final validation results
        % to show in the plot
        ValidationReporter
        
        % WasValidationStopped   (logical) Was training validation stopped?
        WasValidationStopped 
        
        % WasStopButtonPressed   (logical) Was stop button pressed?
        WasStopButtonPressed 
        
        % SummaryFactory   (nnet.internal.cnn.util.SummaryFactory)
        % For creating a summary, purely for the purpose of computing the
        % final validation results.
        SummaryFactory
        
        % EpochInfo   (nnet.internal.cnn.ui.EpochInfo)
        % Contains information to so that the reporter can figure out if we
        % are on the last iteration or not.
        EpochInfo
        
        % LastInfoStruct   (struct) Store of the last infostruct computed
        % so that the presenter can be given it for the post-training stage
        LastInfoStruct = [];
    end
    
    properties(Access = private, Transient)
        % TrainingInterruptEventListener   (listener) Listener on
        % TrainingInterruptEvent fired from ValidationReporter
        TrainingInterruptEventListener
        
        % StopTrainingRequestedListener   (listener) Listener on
        % StopTrainingRequested event fired from TrainingPlotPresenter
        StopTrainingRequestedListener
    end
       
    properties(SetAccess = private, Transient)
        % TrainingPlotPresenter
        % (nnet.internal.cnn.ui.TrainingPlotPresenter)
        % The presenter for the training progress plot.
        TrainingPlotPresenter  
        
        % TrainingStartTime   (datetime) The time that start was called
        TrainingStartTime
    end
    
    methods
        function this = TrainingPlotReporter(trainingPlotPresenter, validationReporter, summaryFactory, epochInfo)
            this.TrainingPlotPresenter = trainingPlotPresenter;
            this.ValidationReporter = validationReporter;
            this.SummaryFactory = summaryFactory;
            this.EpochInfo = epochInfo;
            
            this.WasValidationStopped = false;
            this.WasStopButtonPressed = false;
            
            this.TrainingInterruptEventListener = addlistener(this.ValidationReporter, 'TrainingInterruptEvent', @this.trainingInterruptEventCallback);
            this.StopTrainingRequestedListener = addlistener(this.TrainingPlotPresenter, 'StopTrainingRequested', @this.stopTrainingRequestedCallback);
        end
        
        function setup(this)
            try 
                this.TrainingPlotPresenter.showPreprocessingStage();
            catch ex
                this.cleanUpAfterPlotError(ex);
            end
        end
        
        function start(this)
            try 
                this.TrainingStartTime = datetime('now');
                this.TrainingPlotPresenter.showTrainingStage(this.TrainingStartTime);
            catch ex
                this.cleanUpAfterPlotError(ex);
            end
        end
        
        function reportIteration(this, summary)
            try
                summary.gather();
                infoStruct = iCreateInfoStruct(summary);
                % Only update plot if something has changed
                if ~isequal(infoStruct, this.LastInfoStruct)
                    this.TrainingPlotPresenter.updatePlot(infoStruct);
                end

                this.LastInfoStruct = infoStruct;
            catch ex
                this.cleanUpAfterPlotError(ex); 
            end
        end
       
        function reportEpoch(~,~,~,~)
        end
        
        function finish(this, summary)
            % Incorporate any updates to the summary in the last
            % iteration from other reporters
            
            % finish() is called after the final iteration. Validation
            % results might not have been computed for this final
            % iteration, in which case they'll be computed now. We cannot
            % simply call reportIteration because during-training and
            % end-of-training updates differ.
            try
                summary.gather();
                infoStruct = iCreateInfoStruct(summary);
                % Only update plot if something has changed
                if ~isequal(infoStruct, this.LastInfoStruct)
                    this.TrainingPlotPresenter.updatePlotAtEndOfTraining(infoStruct);
                end
                
                this.LastInfoStruct = infoStruct;
            catch ex
                this.cleanUpAfterPlotError(ex);
            end
        end
        
        function computeFinalValidationResults(this, network)
            try
                % Use validation reporter to compute validation on
                % final network
                finalInfoStruct = this.computeFinalValidation(network);
                stopReason = this.computeStopReason(finalInfoStruct.Iteration);
                this.TrainingPlotPresenter.showPostTrainingStage(this.TrainingStartTime, finalInfoStruct, stopReason);
            catch ex
                this.cleanUpAfterPlotError(ex); 
            end
        end
        
        function finalizePlot(this, errorOccurred)
            this.TrainingPlotPresenter.cleanUpDialogs(); 
            if errorOccurred
                this.TrainingPlotPresenter.displayTrainingErrorMessage(); 
            end
        end
    end
    
    methods(Access = private)
        function cleanUpAfterPlotError(this, exception)
            try
                this.TrainingPlotPresenter.cleanUpDialogs(); 
                this.TrainingPlotPresenter.displayPlotErrorMessage();
                warning(exception.identifier, '%s', exception.message);
            catch
            end
        end
        
        function finalInfoStruct = computeFinalValidation(this, network)
            % compute the final validation results by using the
            % validation reporter.
            summary = this.SummaryFactory.createSummary();
            summary = this.ValidationReporter.computeFinalValidationResultForPlot(summary, network);
            
            % extract the validation results from the summary
            summary.gather();
            finalInfoStruct = this.LastInfoStruct;
            finalInfoStruct.ValidationLoss = summary.ValidationLoss;
            finalInfoStruct.ValidationAccuracy = summary.ValidationAccuracy;
            finalInfoStruct.ValidationRMSE = summary.ValidationRMSE;
        end
        
        function stopReason = computeStopReason(this, iteration)
            if this.reachedMaxIterations(iteration)
                stopReason = nnet.internal.cnn.ui.enum.StopReason.FinalIteration;
            elseif this.WasStopButtonPressed
                stopReason = nnet.internal.cnn.ui.enum.StopReason.StopButton;
            elseif this.WasValidationStopped
                stopReason = nnet.internal.cnn.ui.enum.StopReason.ValidationStopping;
            else
                stopReason = nnet.internal.cnn.ui.enum.StopReason.OutputFcn;
            end
        end
        
        function tf = reachedMaxIterations(this, iteration)
            tf = isequal(this.EpochInfo.NumIters, iteration);
        end
        
        % callbacks
        function trainingInterruptEventCallback(this, ~, ~)
            this.WasValidationStopped = true;
        end
        
        function stopTrainingRequestedCallback(this, ~, ~)
            this.WasStopButtonPressed = true;
            this.notify('TrainingInterruptEvent'); 
        end
    end
end

% helpers
function infoStruct = iCreateInfoStruct(summary)
infoStruct = struct();
infoStruct.Epoch = double(summary.Epoch);
infoStruct.Iteration = double(summary.Iteration);
infoStruct.LearnRate = double(summary.LearnRate);

infoStruct.Loss = double(summary.Loss);
infoStruct.Accuracy = double(summary.Accuracy);
infoStruct.RMSE = double(summary.RMSE);

infoStruct.ValidationLoss = double(summary.ValidationLoss);
infoStruct.ValidationAccuracy = double(summary.ValidationAccuracy);
infoStruct.ValidationRMSE = double(summary.ValidationRMSE);

infoStruct.ElapsedTime = double(summary.Time);
end
