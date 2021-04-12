classdef(Abstract) TableDataFactory < handle
    % TableDataFactory   Interface for factories that create table data
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods(Abstract)
        [sectionName, sectionStructArr] = computeTrainingTimeTableSection(~, trainingStartTime, elapsedTime)
        % computeTrainingTimeTableSection   Computes sectionName and
        % section struct that defines table data for the Training Time
        % section. The section struct array must contain the following
        % fields:
        %  - RowID    (unique id for that row)
        %  - LeftText (text to go in the first column)
        %  - RightText (text to go in the second column)
        
        [sectionName, sectionStructArr] = computeDuringTrainingCycleTableSection(~, epochInfo)
        % computeDuringTrainingCycleTableSection   Computes sectionName and
        % section struct that defines table data for the during-training
        % Training Cycle section. The section struct array must contain the
        % following fields:
        %  - RowID    (unique id for that row)
        %  - LeftText (text to go in the first column)
        %  - RightText (text to go in the second column)
        
        [sectionName, sectionStructArr] = computePostTrainingCycleTableSection(~, finalIteration, finalEpoch, epochInfo)
        % computePostTrainingCycleTableSection   Computes sectionName and
        % section struct that defines table data for the post-training
        % Training Cycle section. The section struct array must contain the
        % following fields:
        %  - RowID    (unique id for that row)
        %  - LeftText (text to go in the first column)
        %  - RightText (text to go in the second column)
        
        [sectionName, sectionStructArr] = computeValidationTableSection(~, validationInfo)
        % computeValidationTableSection   Computes sectionName and section
        % struct that defines table data for the Validation section. The
        % section struct array must contain the following fields:
        %  - RowID    (unique id for that row)
        %  - LeftText (text to go in the first column)
        %  - RightText (text to go in the second column)
        
        [sectionName, sectionStructArr] = computeOtherInfoTableSection(~, executionInfo, learningRate)
        % computeOtherInfoTableSection   Computes sectionName and section
        % struct that defines table data for the Other Information section.
        % The section struct array must contain the following fields:
        %  - RowID    (unique id for that row)
        %  - LeftText (text to go in the first column)
        %  - RightText (text to go in the second column)
        
        
        
        [epochRowID, epochText] = computeEpochTableData(~, epochValue, epochInfo)
        % computeEpochTableData   Given the value of the epoch, get the
        % rowID of the created training table, and a formatted text version
        % of the epochValue.
        
        [elapsedTimeRowID, elapsedTimeText] = computeElapsedTimeTableData(~, elapsedTimeValue)
        % computeElapsedTimeTableData   Given the value of elapsed time, get
        % the rowID of the created training table, and a formatted text
        % version of the elapsed time.
        
        [learnRateRowID, learnRateText] = computeLearningRateTableData(~, learningRateValue)
        % computeLearningRateTableData   Given the value of the learning
        % rate, get the rowID of the created training table, and a
        % formatted text version of the learning rate.
    end
    
end

