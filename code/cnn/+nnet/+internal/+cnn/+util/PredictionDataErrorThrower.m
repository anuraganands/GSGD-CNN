classdef PredictionDataErrorThrower < nnet.internal.cnn.util.ErrorThrowerStrategy
    % PredictionDataErrorThrower   Error thrower strategy for prediction data
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        % Data validation errors
        function throwImageDatastoreMustHaveCategoricalLabels(~)
        end

        function throwImageDatastoreHasNoLabels(~)
        end

        function throwXIsNotValidImageArray(~)
        end

        function throwYIsNotCategoricalResponseVector(~)
        end

        function throwYIsNotValidResponseArray(~)
        end

        function throwXAndYHaveDifferentObservations(~)
        end

        function throwXIsNotValidType(~)
        end

        function throwImageDatastoreWithRegression(~)
        end

        function throwInvalidClassificationTable(~)
        end

        function throwInvalidRegressionTablePredictors(~)
        end

        function throwInvalidRegressionTableResponses(~)
        end

        function throwUndefinedLabels(~)
        end
        
        % Data size validation errors
        function throwOutputSizeNumClassesMismatch(~, ~, ~)
        end

        function throwOutputSizeResponseSizeMismatch(~, ~, ~)
        end

        function throwOutputSizeNumResponsesMismatch(~, ~, ~)
        end
        
        function throwImagesInvalidSize(~, ~, ~)
        end
        
        function throwSequencesInvalidSize(~, DataSize, InputLayerSize)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:PredictionDataErrorThrower:SequencesInvalidSize', DataSize, InputLayerSize);
            throwAsCaller(exception);
        end
        
        function throwXIsNotValidSequenceInput(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:PredictionDataErrorThrower:XIsNotValidSequenceInput');
            throwAsCaller(exception);
        end
        
        function throwYIsNotValidSequenceCategorical(~)
        end
        
        function throwInvalidResponseSequenceLength(~)
        end
        
        function throwOutputModeLastDataMismatchClassification(~)
        end
        
        function throwOutputModeLastDataMismatchRegression(~)
        end
        
        function throwOutputModeSequenceDataMismatchClassification(~)
        end
        
        function throwOutputModeSequenceDataMismatchRegression(~)
        end
        
        function throwIncompatibleInputForRNN(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:PredictionDataErrorThrower:IncompatibleInputForRNN');
            throwAsCaller(exception);
        end
        
        function throwYIsNotValidSequenceResponse(~)
        end
        
        function throwInvalidRNNTablePredictors(~)
        end
        
        function throwInvalidSeq2SeqTableResponse(~)
        end
        
        function throwInvalidSeq2OneTableResponse(~)
        end
        
        function throwOutputModeLastTableDataMismatch(~)
        end
        
        function throwOutputModeSequenceTableDataMismatch(~)
        end
    end
    
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
errorMessage = getString(message(errorID, varargin{:}));
exception = MException(errorID, errorMessage);
end
