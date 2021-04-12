classdef(Abstract) TrainingPlotPresenter < handle
    % TrainingPlotPresenter   Interface for the presenter for the training plot
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    events
        % Event fired when user indicates that they want to stop training
        StopTrainingRequested 
    end
    
    methods(Abstract)
        showPreprocessingStage(this)
        % showPreprocessingStage   Show preprocessing stage of plot.
        
        showTrainingStage(this, trainingStartTime)
        % showTrainingStage   Show training stage of plot.
        
        updatePlot(this, infoStruct)
        % updatePlot   Updates the plot using a struct of information
        
        updatePlotAtEndOfTraining(this, infoStruct)
        % updatePlotAtEndOfTraining   Updates the plot using a struct of
        % information at the end of training.
        
        showPostTrainingStage(this, trainingStartTime, infoStruct, stopReason)
        % showPostTrainingStage   Show the post-training stage of the plot
        
        cleanUpDialogs(this)
        % cleanUpDialogs   Clean up dialogs
        
        displayTrainingErrorMessage(this)
        % displayTrainingErrorMessage   Displays message explaining that
        % error occurred during training.
        
        displayPlotErrorMessage(test)
        % displayPlotErrorMessage   Displays message explaining that error
        % occurred in plot.
    end  
end
