classdef (Abstract) ErrorThrowerStrategy
    % ErrorThrowerStrategy   Error thrower strategy interface
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods (Abstract)
        % Data validation errors
        throwImageDatastoreHasNoLabels(this)
        throwImageDatastoreMustHaveCategoricalLabels(this)
        throwXIsNotValidImageArray(this)
        throwYIsNotCategoricalResponseVector(this)
        throwYIsNotValidResponseArray(this)
        throwXAndYHaveDifferentObservations(this)
        throwXIsNotValidType(this)
        throwImageDatastoreWithRegression(this)
        throwInvalidClassificationTable(this)
        throwInvalidRegressionTablePredictors(this)
        throwInvalidRegressionTableResponses(this)
        throwUndefinedLabels(this)
        throwXIsNotValidSequenceInput(this)
        throwYIsNotValidSequenceCategorical(this)
        throwInvalidResponseSequenceLength(this)
        throwOutputModeLastDataMismatchClassification(this)
        throwOutputModeSequenceDataMismatchClassification(this)
        throwOutputModeLastDataMismatchRegression(this)
        throwOutputModeSequenceDataMismatchRegression(this)
        throwIncompatibleInputForRNN(this)
        throwYIsNotValidSequenceResponse(this)
        throwInvalidRNNTablePredictors(this)
        throwInvalidSeq2OneTableResponse(this)
        throwInvalidSeq2SeqTableResponse(this)
        throwOutputModeLastTableDataMismatch(this)
        throwOutputModeSequenceTableDataMismatch(this)
        
        % Data size validation errors
        
        % throwOutputSizeNumClassesMismatch
        % Input:
        %    networkOutputSize (char vector)   Network output size
        %    dataNumClasses (char vector)      Number of classes in the data
        throwOutputSizeNumClassesMismatch(this, networkOutputSize, dataNumClasses)
        
        % throwOutputSizeResponseSizeMismatch
        % Input:
        %    networkOutputSize (char vector)   Network output size
        %    dataResponseSize (char vector)    Response size from data
        throwOutputSizeResponseSizeMismatch(this, networkOutputSize, dataResponseSize)
        
        % throwOutputSizeResponseSizeMismatch
        % Input:
        %    networkOutputSize (char vector)   Network output size
        %    dataNumResponses (char vector)    Number of response in the data
        throwOutputSizeNumResponsesMismatch(this, networkOutputSize, dataNumResponses)
        
        % throwOutputSizeResponseSizeMismatch
        % Input:
        %    ImageSize (char vector)   Input image size
        %    InputSize (char vector)   Input layer size
        throwImagesInvalidSize(this, ImageSize, InputLayerSize)
        
        % throwSequencesInvalidSize
        % Input:
        %    DataSize (integer)         Sequence data dimension
        %    InputLayerSize (integer)   Input layer size
        throwSequencesInvalidSize(this, DataSize, InputLayerSize)
    end
    
end