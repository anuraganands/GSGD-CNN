classdef ValidationDataErrorThrower < nnet.internal.cnn.util.ErrorThrowerStrategy
    % ValidationDataErrorThrower   Error thrower strategy for validation data
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        % Data validation errors
        function throwImageDatastoreMustHaveCategoricalLabels(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:ImageDatastoreMustHaveCategoricalLabels');
            throwAsCaller(exception);
        end

        function throwImageDatastoreHasNoLabels(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:ImageDatastoreHasNoLabels');
            throwAsCaller(exception);
        end

        function throwXIsNotValidImageArray(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:XIsNotValidImageArray');
            throwAsCaller(exception);
        end

        function throwYIsNotCategoricalResponseVector(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:YIsNotCategoricalResponseVector');
            throwAsCaller(exception);
        end

        function throwYIsNotValidResponseArray(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:YIsNotValidResponseArray');
            throwAsCaller(exception);
        end

        function throwXAndYHaveDifferentObservations(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:XAndYHaveDifferentObservations');
            throwAsCaller(exception);
        end

        function throwXIsNotValidType(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:XIsNotValidType');
            throwAsCaller(exception);
        end

        function throwImageDatastoreWithRegression(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:ImageDatastoreWithRegression');
            throwAsCaller(exception);
        end

        function throwInvalidClassificationTable(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:InvalidClassificationTable');
            throwAsCaller(exception);
        end

        function throwInvalidRegressionTablePredictors(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:InvalidRegressionTablePredictors');
            throwAsCaller(exception);
        end

        function throwInvalidRegressionTableResponses(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:InvalidRegressionTableResponses');
            throwAsCaller(exception);
        end

        function throwUndefinedLabels(~)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:UndefinedLabels');
            throwAsCaller(exception);
        end
        
        % Data size validation errors
        function throwOutputSizeNumClassesMismatch(~, networkOutputSize, dataNumClasses)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:OutputSizeNumClassesMismatch', networkOutputSize, dataNumClasses);
            throwAsCaller(exception);
        end

        function throwOutputSizeResponseSizeMismatch(~, networkOutputSize, dataResponseSize)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:OutputSizeResponseSizeMismatch', networkOutputSize, dataResponseSize);
            throwAsCaller(exception);
        end

        function throwOutputSizeNumResponsesMismatch(~, networkOutputSize, dataNumResponses)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:OutputSizeNumResponsesMismatch', networkOutputSize, dataNumResponses);
            throwAsCaller(exception);
        end
        
        function throwImagesInvalidSize(~, ImageSize, InputLayerSize)
            exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:util:ValidationDataErrorThrower:ImagesInvalidSize', ImageSize, InputLayerSize);
            throwAsCaller(exception);
        end
        
        function throwSequencesInvalidSize(~, ~, ~)
        end
        
        function throwXIsNotValidSequenceInput(~)
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
