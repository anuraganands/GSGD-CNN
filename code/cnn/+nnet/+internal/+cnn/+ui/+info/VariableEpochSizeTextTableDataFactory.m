classdef VariableEpochSizeTextTableDataFactory < nnet.internal.cnn.ui.info.TableDataFactory
    % VariableEpochSizeTextTableDataFactory   Factory for the TextTable that goes in
    % the infostrip of the TrainingPlotView when we have variable number of
    % iterations per epoch.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function [sectionName, sectionStructArr] = computeTrainingTimeTableSection(~, trainingStartTime, elapsedTime)
             sectionName = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripTrainingTimeSectionName');
             sectionStructArr = struct();
             
             % Start time
             sectionStructArr(1).RowID = 'TRAINING_START_TIME';
             sectionStructArr(1).LeftText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripTrainingStartTimeLabel');
             sectionStructArr(1).RightText = iDateTimeAsStringUsingDefaultLocalFormat(trainingStartTime);
             % Elapsed time
             sectionStructArr(2).RowID = iElapsedTimeRowID();
             sectionStructArr(2).LeftText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripTrainingElapsedTimeLabel');
             sectionStructArr(2).RightText = iFormattedElapsedTimeValue(elapsedTime);
        end
        
        function [sectionName, sectionStructArr] = computeDuringTrainingCycleTableSection(~, epochInfo)
            sectionName = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripTrainingCycleSectionName');
            sectionStructArr = struct();
            
            % Epoch
            currEpoch = 0;
            sectionStructArr(1).RowID = iEpochRowID();
            sectionStructArr(1).LeftText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripEpochLabel');
            sectionStructArr(1).RightText = iFormattedEpochValue(currEpoch, epochInfo);
        end
        
        function [sectionName, sectionStructArr] = computePostTrainingCycleTableSection(~, finalIteration, finalEpoch, epochInfo)
            sectionName = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripTrainingCycleSectionName');
            sectionStructArr = struct();
            
            % Epoch
            sectionStructArr(1).RowID = iEpochRowID();
            sectionStructArr(1).LeftText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripEpochLabel');
            sectionStructArr(1).RightText = iFormattedEpochValue(finalEpoch, epochInfo);
            % Final iteration
            sectionStructArr(2).RowID = iFinalIterationRowID();
            sectionStructArr(2).LeftText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripFinalIterationLabel');
            sectionStructArr(2).RightText = iFormattedFinalIteration(finalIteration);
        end
        
        function [sectionName, sectionStructArr] = computeValidationTableSection(~, validationInfo)
            sectionName = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripValidationSectionName');
            sectionStructArr = struct();
            
            % Validation Frequency
            sectionStructArr(1).RowID = 'VALIDATION_FREQUENCY';
            sectionStructArr(1).LeftText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripValidationFrequencyLabel');
            sectionStructArr(1).RightText = iComputeValidationFrequencyValue(validationInfo);
            % Validation Patience
            sectionStructArr(2).RowID = 'VALIDATION_PATIENCE';
            sectionStructArr(2).LeftText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripValidationPatienceLabel');
            sectionStructArr(2).RightText = iComputeValidationPatienceValue(validationInfo);
        end
        
        function [sectionName, sectionStructArr] = computeOtherInfoTableSection(~, executionInfo, learningRate)
            sectionName = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripOtherInformationSectionName');
            sectionStructArr = struct();
            
            % Hardware resource
            sectionStructArr(1).RowID = 'HARDWARE_RESOURCE';
            sectionStructArr(1).LeftText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripHardwareResourceLabel');
            sectionStructArr(1).RightText = iComputeHardwareResourceValue(executionInfo);
            % Learning rate schedule
            sectionStructArr(2).RowID = 'LEARNING_RATE_SCHEDULE';
            sectionStructArr(2).LeftText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripLearningRateScheduleLabel');
            sectionStructArr(2).RightText = iComputeLearningRateScheduleValue(executionInfo);
            % Learning rate
            sectionStructArr(3).RowID = iLearningRateRowID();
            sectionStructArr(3).LeftText = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripLearningRateLabel');
            sectionStructArr(3).RightText = iFormattedLearningRate(learningRate);
        end
        
        function [epochRowID, formattedEpoch] = computeEpochTableData(~, epochValue, epochInfo)
            epochRowID = iEpochRowID();
            formattedEpoch = iFormattedEpochValue(epochValue, epochInfo);
        end
        
        function [elapsedTimeRowID, formattedElapsedTime] = computeElapsedTimeTableData(~, elapsedTimeValue)
            elapsedTimeRowID = iElapsedTimeRowID();
            formattedElapsedTime = iFormattedElapsedTimeValue(elapsedTimeValue);
        end
        
        function [learningRateRowID, formattedLearningRate] = computeLearningRateTableData(~, learningRate)
            learningRateRowID = iLearningRateRowID();
            formattedLearningRate = iFormattedLearningRate(learningRate);
        end
    end
end

% helpers
function messageString = iMessageString(id, varargin)
m = message(id, varargin{:});
messageString = m.getString();
end

function str = iDateTimeAsStringUsingDefaultLocalFormat(dt)
defaultFormat = datetime().Format;
dt.Format = defaultFormat;
str = char(dt);
end

function rowID = iEpochRowID()
rowID = 'EPOCH';
end

function rowID = iFinalIterationRowID()
rowID = 'FINAL_ITERATION';
end

function rowID = iElapsedTimeRowID()
rowID = 'TRAINING_ELAPSED_TIME';
end

function rowID = iLearningRateRowID()
rowID = 'LEARNING_RATE';
end

function str = iFormattedEpochValue(currEpoch, epochInfo)
str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripEpochValue', num2str(currEpoch), num2str(epochInfo.NumEpochs));
end

function str = iFormattedFinalIteration(finalIteration)
str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripFinalIterationValueWithoutMaxIters', num2str(finalIteration));
end

function str = iFormattedElapsedTimeValue(elapsedTimeInSecs)
[fullMins, remainingSecs] = iMinutes(elapsedTimeInSecs);
roundedSecs = floor(remainingSecs);
if fullMins == 0
    str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripTrainingElapsedTimeValueInSecs', num2str(roundedSecs));
else
    str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripTrainingElapsedTimeValueInMins', num2str(fullMins), num2str(roundedSecs));
end
end

function [fullMins, remainingSecs] = iMinutes(elapsedTimeInSecs)
numMins = elapsedTimeInSecs / 60;
fullMins = floor(numMins);
remainingSecs = elapsedTimeInSecs - fullMins * 60;
end

function str = iFormattedLearningRate(learningRate)
str = num2str(learningRate, 5);
end

function str = iComputeValidationFrequencyValue(validationInfo)
if validationInfo.IsValidationEnabled
    str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripValidationFrequencyValue', validationInfo.ValidationFrequency);
else
    str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripValidationFrequencyValueNotUsed');
end
end

function str = iComputeValidationPatienceValue(validationInfo)
if validationInfo.IsValidationEnabled
    str = num2str(validationInfo.ValidationPatience);
else
    str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripValidationPatienceValueNotUsed');
end
end

function str = iComputeHardwareResourceValue(executionInfo)
switch executionInfo.ExecutionEnvironment
    case 'cpu'
        if executionInfo.UseParallel
            str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripHardwareResourceValueParallelCPU');
        else
            str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripHardwareResourceValueSerialCPU');
        end
    case 'gpu'
        if executionInfo.UseParallel
            str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripHardwareResourceValueParallelGPU');
        else
            str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripHardwareResourceValueSerialGPU');
        end
    otherwise
        warning(message('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripInvalidValue'));
        str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripHardwareResourceValueSerialGPU');
end
end

function str = iComputeLearningRateScheduleValue(executionInfo)
switch executionInfo.LearnRateSchedule
    case 'none'
        str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripLearningRateScheduleConstant');
    case 'piecewise'
        str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripLearningRateSchedulePiecewise');
    otherwise
        warning(message('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripInvalidValue'));
        str = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:InfoStripLearningRateScheduleConstant');
end
end
