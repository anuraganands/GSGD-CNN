classdef TrainingPlotPresenterWithDialog < nnet.internal.cnn.ui.TrainingPlotPresenter
    % TrainingPlotPresenterWithDialog   Presenter for the training plot with a dialog for initialization.
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % TrainingPlotView   (nnet.internal.cnn.ui.TrainingPlotView)
        % The view object
        TrainingPlotView
        
        % PreprocessingDisplayer   (nnet.internal.cnn.ui.PreprocessingDisplayer)
        % For displaying preprocessing information
        PreprocessingDisplayer
    end
    
    properties(Access = private)        
        % DialogFactory   (nnet.internal.cnn.ui.DialogFactory) Creates
        % closing dialog
        DialogFactory
        
        % AxesFactory   (nnet.internal.cnn.ui.AxesFactory)
        % Factory for creating AxesViews.
        AxesFactory
        
        % TableDataFactory  (nnet.internal.cnn.ui.info.TableDataFactory)
        % Factory for setting table of information in the view
        TableDataFactory
        
        % MetricRowDataFactory   (nnet.internal.cnn.ui.info.MetricRowDataFactory)
        % Factory for setting the table-row for the final validation result
        MetricRowDataFactory
        
        % StopReasonRowDataFactory   (nnet.internal.cnn.ui.info.StopReasonRowDataFactory)
        % Factory for setting the table-row for the stop reason
        StopReasonRowDataFactory
        
        % EpochDisplayer   (nnet.internal.cnn.ui.axes.EpochDisplayer) For
        % displaying the epoch information in the various axes.
        EpochDisplayer
        
        % HelpLauncher   (nnet.internal.cnn.ui.info.HelpLauncher)
        HelpLauncher
        
        % Watch   (nnet.internal.cnn.ui.adapter.Watch)
        Watch
        
        % ExecutionInfo   (nnet.internal.cnn.ui.ExecutionInfo) The
        % evaluated execution settings
        ExecutionInfo
        
        % ValidationInfo   (nnet.internal.cnn.ui.ValidationInfo)
        % The validation info
        ValidationInfo
        
        % EpochInfo   (nnet.internal.cnn.ui.EpochInfo) Information relating
        % to epochs and iterations.
        EpochInfo
        
        % UpdateableMetrics   (cell of nnet.internal.cnn.ui.metric.UpdateableMetric)
        UpdateableMetrics
        
        % AxesViews   (cell of nnet.internal.cnn.ui.axes.AxesView)
        AxesViews

        % AllowUserToCloseFigure   (logical) Allow user to close the figure
        AllowUserToCloseFigure
        
        % MessageBox   (figure) Message box warning users that they cannot
        % close the main plot figure during training
        MessageBox
        
        % HelpLinkClickedListener   (listener) Listener for the
        % HelpLinkClicked event on TrainingPlotView
        HelpLinkClickedListener
        
        % StopButtonClickedListener   (listener) Listener to the
        % StopButtonClicked event on TrainingPlotView
        StopButtonClickedListener
        
        % FigureCloseRequestedListener   (listener) Listener for the
        % FigureCloseRequestedListener event on TrainingPlotView
        FigureCloseRequestedListener
    end
    
    methods
        function this = TrainingPlotPresenterWithDialog(...
                trainingPlotView, tableDataFactory, metricRowDataFactory, stopReasonRowDataFactory, preprocessingDisplayer, dialogFactory, ...
                axesFactory, epochDisplayer, helpLauncher, watch, executionInfo, validationInfo, epochInfo)
            this.TrainingPlotView = trainingPlotView;            
            this.TableDataFactory = tableDataFactory;
            this.MetricRowDataFactory = metricRowDataFactory;
            this.StopReasonRowDataFactory = stopReasonRowDataFactory;
            this.PreprocessingDisplayer = preprocessingDisplayer;
            this.DialogFactory = dialogFactory;
            this.AxesFactory = axesFactory;
            this.EpochDisplayer = epochDisplayer;
            this.HelpLauncher = helpLauncher;
            this.Watch = watch;
            this.ExecutionInfo = executionInfo;
            this.ValidationInfo = validationInfo;
            this.EpochInfo = epochInfo;
            
            this.AllowUserToCloseFigure = false;
            this.setupView();
            this.HelpLinkClickedListener = addlistener(this.TrainingPlotView, 'HelpLinkClicked', @this.helpLinkClickedCallback);
            this.StopButtonClickedListener = addlistener(this.TrainingPlotView, 'StopButtonClicked', @this.stopButtonClickedCallback);
            this.FigureCloseRequestedListener = addlistener(this.TrainingPlotView, 'FigureCloseRequested', @this.figureCloseRequestedCallback);
            
            this.wrapThisAndGiveToView();
        end

        function showPreprocessingStage(this)
            this.PreprocessingDisplayer.displayPreprocessing(this.TrainingPlotView);
        end
        
        function showTrainingStage(this, trainingStartTime) 
            this.PreprocessingDisplayer.hidePreprocessing(this.TrainingPlotView);
            
            this.TrainingPlotView.setTitleInfo(trainingStartTime);
            this.setupDuringTrainingTable(trainingStartTime);
            
            this.TrainingPlotView.showMainPlot();
            
            this.Watch.reset();
        end
        
        function updatePlot(this, infoStruct)
            this.updateProgress(infoStruct);
            this.updateDuringTrainingTable(infoStruct);
            this.updateMetricsDuringTraining(infoStruct);
            this.updateAxesEvery( iMinUpdatePeriodInSeconds() );
        end
        
        function updatePlotAtEndOfTraining(this, infoStruct)
            % Called when the Reporter is in the finish() stage of
            % training.
            
            this.updateProgress(infoStruct);
            this.updateDuringTrainingTable(infoStruct);
            this.updateMetricsAtEndOfTraining(infoStruct);
            this.updateAxes();
            
            drawnow();
        end
        
        function showPostTrainingStage(this, trainingStartTime, infoStruct, stopReason)
            this.AllowUserToCloseFigure = true;
            
            this.TrainingPlotView.ProgressPanelVisible = false;
            this.setupPostTrainingTable(trainingStartTime, infoStruct, stopReason);
            
            this.updateMetricsAtEndOfTraining(infoStruct);
            this.finalizeAxes();
            
            drawnow();
        end
        
        function cleanUpDialogs(this)
            this.PreprocessingDisplayer.hidePreprocessing(this.TrainingPlotView);
            delete(this.MessageBox);
            this.AllowUserToCloseFigure = true;
            if ~this.TrainingPlotView.FigureExistsAndIsVisible
                this.TrainingPlotView.closeFigure();
            end
        end
        
        function displayTrainingErrorMessage(this)
            if this.TrainingPlotView.FigureExistsAndIsVisible
                this.showOnlyTrainingErrorMessage()
            end
        end
         
        function displayPlotErrorMessage(this)
            % Training error message takes priority over plot error message
            % (only one of the two messages should be displayed at once).
            if this.TrainingPlotView.FigureExistsAndIsVisible && ~this.TrainingPlotView.TrainingErrorMessageVisible
                this.TrainingPlotView.PlotErrorMessageVisible = true;
            end
        end
    end
    
    methods(Access = private)
        function setupView(this)
            this.setupAxesAndMetrics();
            this.setupProgress();
            this.setupLegend();
        end
        
        function setupAxesAndMetrics(this)
            [this.AxesViews{1}, metrics1] = this.AxesFactory.createMainAxesAndMetrics(this.EpochInfo, this.EpochDisplayer);
            [this.AxesViews{2}, metrics2] = this.AxesFactory.createLossAxesAndMetrics(this.EpochInfo, this.EpochDisplayer);
            
            this.TrainingPlotView.setTopAxes(this.AxesViews{1});
            this.TrainingPlotView.setBottomAxes(this.AxesViews{2});
            
            this.UpdateableMetrics = [metrics1, metrics2];
        end
        
        function setupProgress(this)
            currValue = 0;
            maxValue = this.EpochInfo.NumIters;
            this.TrainingPlotView.setupProgress(currValue, maxValue); 
        end
        
        function setupLegend(this)
            [cellOfLegendSectionNames, cellOfLegendSectionStructArrs] = this.AxesFactory.createLegendInfo();
            
            areCheckboxesVisible = false;
            this.TrainingPlotView.setupLegend(cellOfLegendSectionNames, cellOfLegendSectionStructArrs, areCheckboxesVisible);
        end
        
        function updateMetricsDuringTraining(this, infoStruct)
            cellfun(@(m) m.update(infoStruct), this.UpdateableMetrics); 
        end
        
        function updateMetricsAtEndOfTraining(this, infoStruct)
            cellfun(@(m) m.updatePostTraining(infoStruct), this.UpdateableMetrics); 
        end
        
        function updateAxesEvery(this, periodInSeconds)
            durationSinceLastReset = this.Watch.getDurationSinceReset();
            if durationSinceLastReset > periodInSeconds
                this.updateAxes();
                drawnow();
                this.Watch.reset();  
            end
        end
        
        function updateAxes(this)
            for i=1:numel(this.AxesViews)
                this.AxesViews{i}.update();
            end
        end
        
        function updateProgress(this, infoStruct)
            this.TrainingPlotView.updateProgress( infoStruct.Iteration ); 
        end
             
        function finalizeAxes(this)
            for i=1:numel(this.AxesViews)
                this.AxesViews{i}.finalize();
            end
        end
        
        function setupDuringTrainingTable(this, trainingStartTime)
            tableSectionNames = {};
            tableSectionStructs = {};
            elapsedTime = 0;
            learningRate = this.ExecutionInfo.InitialLearningRate;
            [tableSectionNames{1}, tableSectionStructs{1}] = this.TableDataFactory.computeTrainingTimeTableSection(trainingStartTime, elapsedTime);
            [tableSectionNames{2}, tableSectionStructs{2}] = this.TableDataFactory.computeDuringTrainingCycleTableSection(this.EpochInfo);
            [tableSectionNames{3}, tableSectionStructs{3}] = this.TableDataFactory.computeValidationTableSection(this.ValidationInfo);
            [tableSectionNames{4}, tableSectionStructs{4}] = this.TableDataFactory.computeOtherInfoTableSection(this.ExecutionInfo, learningRate);
            
            this.TrainingPlotView.setupTableOfData(tableSectionNames, tableSectionStructs);
        end
        
        function setupPostTrainingTable(this, trainingStartTime, infoStruct, stopReason)
            tableSectionNames = {};
            tableSectionStructs = {};
            [tableSectionNames{1}, tableSectionStructs{1}] = this.createResultsTableSection(infoStruct, stopReason);
            [tableSectionNames{2}, tableSectionStructs{2}] = this.TableDataFactory.computeTrainingTimeTableSection(trainingStartTime, infoStruct.ElapsedTime);
            [tableSectionNames{3}, tableSectionStructs{3}] = this.TableDataFactory.computePostTrainingCycleTableSection(infoStruct.Iteration, infoStruct.Epoch, this.EpochInfo);
            [tableSectionNames{4}, tableSectionStructs{4}] = this.TableDataFactory.computeValidationTableSection(this.ValidationInfo);
            [tableSectionNames{5}, tableSectionStructs{5}] = this.TableDataFactory.computeOtherInfoTableSection(this.ExecutionInfo, infoStruct.LearnRate);
            
            this.TrainingPlotView.setupTableOfData(tableSectionNames, tableSectionStructs);
        end
        
        function updateDuringTrainingTable(this, infoStruct)
            [epochRowID, formattedEpoch] = this.TableDataFactory.computeEpochTableData(infoStruct.Epoch, this.EpochInfo);
            [elapsedTimeRowID, formattedElapsedTime] = this.TableDataFactory.computeElapsedTimeTableData(infoStruct.ElapsedTime);
            [learningRateRowID, formattedLearningRate] = this.TableDataFactory.computeLearningRateTableData(infoStruct.LearnRate);
            
            this.TrainingPlotView.updateTableOfData(epochRowID, formattedEpoch);
            this.TrainingPlotView.updateTableOfData(elapsedTimeRowID, formattedElapsedTime);
            this.TrainingPlotView.updateTableOfData(learningRateRowID, formattedLearningRate);
        end
        
        function [tableSectionName, tableSectionStruct] = createResultsTableSection(this, infoStruct, stopReason)
            tableSectionName = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripResultsSectionName');
            
            metricRowStruct = this.MetricRowDataFactory.createMetricRowData(infoStruct);
            stopReasonRowStruct = this.StopReasonRowDataFactory.createStopReasonRowData(stopReason);
            tableSectionStruct = [metricRowStruct, stopReasonRowStruct];
        end
        
        function wrapThisAndGiveToView(this)
            encapsulatingWrapper = nnet.internal.cnn.ui.EncapsulatingWrapper(this);
            this.TrainingPlotView.setUserData(encapsulatingWrapper);
        end
        
        function showOnlyTrainingErrorMessage(this)
            this.TrainingPlotView.PlotErrorMessageVisible = false;
            this.TrainingPlotView.TrainingErrorMessageVisible = true;
        end
        
        % callbacks
        function helpLinkClickedCallback(this, ~, ~)
            this.HelpLauncher.openLearnMoreHelpPage();
        end
        
        function stopButtonClickedCallback(this, ~, ~)
            this.notify('StopTrainingRequested'); 
        end
        
        function figureCloseRequestedCallback(this, ~, ~)
            if this.AllowUserToCloseFigure
                this.TrainingPlotView.closeFigure(); 
            else
                mainMessage = iMessage('nnet_cnn:internal:cnn:ui:trainingplot:CannotClosePlotDuringTrainingMessage');
                titleMessage = iMessage('nnet_cnn:internal:cnn:ui:trainingplot:CannotClosePlotDuringTrainingTitleMessage');
                this.MessageBox = this.DialogFactory.createMessageBox(mainMessage, titleMessage, 'NNET_CNN_TRAININGPLOT_CANNOTCLOSINGDURINGTRAININGDIALOG');
            end
        end
    end    
end

% helpers
function m = iMessage(varargin)
m = message(varargin{:});
end

function str = iMessageString(varargin)
m = message(varargin{:});
str = m.getString();
end

function d = iMinUpdatePeriodInSeconds()
d = 0.21 * seconds;
end
