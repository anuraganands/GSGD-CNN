classdef(Abstract) Summary < handle
    % Summary   Interface for holding training summary information
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Abstract)
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
        
        % Predictions   4-D array of network predictions
        Predictions
        
        % Response   4-D array of responses
        Response
        
        % ValidationPredictions   4-D array of validation predictions
        ValidationPredictions
        
        % ValidationResponse   4-D array of validation responses
        ValidationResponse
    end
    
    properties (Abstract, SetAccess = protected)  
        % Accuracy (double)   Current accuracy for a classification problem
        Accuracy
        
        % RMSE (double)   Current RMSE for a regression problem
        RMSE 
        
        % ValidationAccuracy (double)   Current validation accuracy for a
        % classification problem
        ValidationAccuracy
        
        % ValidationRMSE (double)   Current validation RMSE for a
        % regression problem
        ValidationRMSE 
    end
    
    methods(Abstract)
        update( this, predictions, response, epoch, iteration, elapsedTime, miniBatchLoss, learnRate )
        % update   Use this function to update all the
        % properties of the class without having to individually fill
        % in each property.
        
        gather( this )
        % gather  Ensure all properties are stored on the host
    end
    
end

