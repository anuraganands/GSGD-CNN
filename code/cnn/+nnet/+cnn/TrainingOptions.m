classdef(Abstract) TrainingOptions
    % TrainingOptions   Base training options for stochastic solvers.
    %
    %   This class holds the training options common to stochastic solvers.
    %
    %   TrainingOptions properties:
    %       LearnRateScheduleSettings   - Settings for the learning rate
    %                                     schedule.
    %       L2Regularization            - Factor for L2 regularization.
    %       GradientThresholdMethod     - Method for gradient thresholding.
    %       GradientThreshold           - Gradient threshold.
    %       MaxEpochs                   - Maximum number of epochs.
    %       MiniBatchSize               - The size of a mini-batch for
    %                                     training.
    %       Verbose                     - Flag for printing information to
    %                                     the command window.
    %       VerboseFrequency            - This only has an effect if
    %                                     'Verbose' is set to true. It
    %                                     specifies the number of
    %                                     iterations between printing to
    %                                     the command window.
    %       ValidationData              - Data to use for validation during
    %                                     training.
    %       ValidationFrequency         - Number of iterations between
    %                                     evaluations of validation
    %                                     metrics.
    %       ValidationPatience          - The number of times that the
    %                                     validation loss is allowed to be
    %                                     larger than or equal to the
    %                                     previously smallest loss before
    %                                     training is stopped.
    %       Shuffle                     - This controls if the training
    %                                     data is shuffled.
    %       CheckpointPath              - Path where checkpoint networks
    %                                     will be saved.
    %       ExecutionEnvironment        - What hardware to use for training
    %                                     the network.
    %       WorkerLoad                  - Specify compute and prefetch
    %                                     workers and their relative load
    %                                     in a parallel pool.
    %       OutputFcn                   - User callback to be executed at
    %                                     each iteration.
    %       Plots                       - Plots to display during training
    %       SequenceLength              - Sequence length of a mini-batch
    %                                     during training.
    %       SequencePaddingValue        - Value to pad mini-batches along
    %                                     the sequence dimension.
    %       isGuided                    - Executes Training with a Guided
    %                                     Approach to Weight Update with live identification
    %                                     of consistent datasets
    %       Rho                         - number of iterations of data
    %                                     collection for assessment
    %                                     and identification of consistent data from inconsistent data before
    %                                     guided weight update is executed
    %       RevisitBatchNum             - number of batches to revisit
    %                                     to assess for consistentcy of
    %                                     previous batches on current batch
    %                                     weights
    
    %       VerificationSetNum          - number of batches to set aside in
    %                                     every epoch for calculation of
    %                                     true error or grdient of training data
    
    % Copyright 2017 The MathWorks, Inc.
    
    properties(Abstract,Access = protected)
        % Version   Number to identify the current version of this object
        %   This is used to ensure that objects from older versions are
        %   loaded correctly.
        Version
    end
    
    properties(SetAccess = private)
        % LearnRateScheduleSettings   Settings for the learning rate schedule
        %   The learning rate schedule settings. This summarizes the
        %   options for the chosen learning rate schedule. The field Method
        %   gives the name of the method for adjusting the learning rate.
        %   This can either be 'none', in which case the learning rate is
        %   not altered, or 'piecewise', in which case there will be two
        %   additional fields. These fields are DropFactor, which is a
        %   multiplicative factor for dropping the learning rate, and
        %   DropPeriod, which determines how many epochs should pass
        %   before dropping the learning rate.
        LearnRateScheduleSettings
        
        % L2Regularization   Factor for L2 regularization
        %   The factor for the L2 regularizer. It should be noted that each
        %   set of parameters in a layer can specify a multiplier for this
        %   L2 regularizer.
        L2Regularization
        
        % GradientThresholdMethod   Method used for gradient thresholding
        %   Method used for thresholding the gradient. Options are
        %   'global-l2norm', 'l2norm' (default) and 'absolute-value'.
        GradientThresholdMethod
        
        % GradientThreshold   Threshold used for gradients
        %   The threshold used to scale the gradients by the method
        %   specified in GradientThresholdMethod.
        GradientThreshold
        
        % MaxEpochs   Maximum number of epochs
        %   The maximum number of epochs that will be used for training.
        %   Training will stop once this number of epochs has passed.
        MaxEpochs
        
        % MiniBatchSize   The size of a mini-batch for training
        %   The size of the mini-batch used for each training iteration.
        MiniBatchSize
        
        % Verbose   Flag for printing information to the command window
        %   If this is set to true, information on training progress will
        %   be printed to the command window. The default is true.
        Verbose
        
        % VerboseFrequency   Frequency for printing information
        %   This only has an effect if 'Verbose' is true. It specifies the
        %   number of iterations between printing to the command window.
        VerboseFrequency
        
        % ValidationData   Data to be used for validation purposes
        ValidationData
        
        % ValidationFrequency   Frequency for computing validation metrics
        ValidationFrequency
        
        % ValidationPatience   Patience used to stop training
        ValidationPatience
        
        % Shuffle   This controls when the training data is shuffled. It
        % can either be 'once' to shuffle data once before training,
        % 'every-epoch' to shuffle before every training epoch, or 'never'
        % in order not to shuffle the data.
        Shuffle
        
        % CheckpointPath   This is the path where the checkpoint networks
        % will be saved. If empty, no checkpoint will be saved.
        CheckpointPath
        
        % ExecutionEnvironment   Determines what hardware to use for
        % training the network.
        ExecutionEnvironment
        
        % WorkerLoad   Relative division of load between parallel workers
        % on different hardware
        WorkerLoad
        
        % OutputFcn   Functions to call after each iteration, passing
        % training info from the current iteration, and returning true to
        % terminate training early
        OutputFcn
        
        % Plots   Plots to show during training
        Plots
        
        % SequenceLength   Determines the strategy used to create
        % mini-batches of sequence data
        SequenceLength
        
        % SequencePaddingValue   Scalar value used to pad mini-batches in
        % the along the sequence dimension
        SequencePaddingValue
        
        %Executes Training with a Guided
        %Approach to Weight Update with live identification
        %of consistent datasets
        isGuided
        
        %number of iterations of data
        %collection for assessment
        %and identification of consistent data from inconsistent data before
        %guided weight update is executed
        Rho 
        
        %number of batches to revisit
        %to assess for consistentcy of
        %previous batches on current batch
        %weights
        RevisitBatchNum 
    
        %number of batches to set aside in
        %every epoch for calculation of
        %true error or grdient of training data
        VerificationSetNum
        
    end
    
    methods(Access = public)
        function this = TrainingOptions(inputArguments)
            this.LearnRateScheduleSettings = iCreateLearnRateScheduleSettings( ...
                inputArguments.LearnRateSchedule, ...
                inputArguments.LearnRateDropFactor, ...
                inputArguments.LearnRateDropPeriod);
            this.L2Regularization = inputArguments.L2Regularization;
            this.GradientThresholdMethod = inputArguments.GradientThresholdMethod;
            this.GradientThreshold = inputArguments.GradientThreshold;
            this.MaxEpochs = inputArguments.MaxEpochs;
            this.MiniBatchSize = inputArguments.MiniBatchSize;
            this.Verbose = inputArguments.Verbose;
            this.VerboseFrequency = inputArguments.VerboseFrequency;
            this.ValidationData = inputArguments.ValidationData;
            this.ValidationFrequency = inputArguments.ValidationFrequency;
            this.ValidationPatience = inputArguments.ValidationPatience;
            this.Shuffle = inputArguments.Shuffle;
            this.CheckpointPath = inputArguments.CheckpointPath;
            this.ExecutionEnvironment = inputArguments.ExecutionEnvironment;
            this.WorkerLoad = inputArguments.WorkerLoad;
            this.OutputFcn = inputArguments.OutputFcn;
            this.Plots = inputArguments.Plots;
            this.SequenceLength = inputArguments.SequenceLength;
            this.SequencePaddingValue = inputArguments.SequencePaddingValue;
            this.isGuided = inputArguments.isGuided;
            this.Rho = inputArguments.Rho; 
            this.RevisitBatchNum = inputArguments.RevisitBatchNum;
            this.VerificationSetNum = inputArguments.VerificationSetNum;
        end
        
        function out = saveobj(this)
            out.Version = this.Version;
            out = nnet.cnn.TrainingOptions.flattenLearnRateScheduleSettings(out, this.LearnRateScheduleSettings);
            out.L2Regularization = this.L2Regularization;
            out.GradientThresholdMethod = this.GradientThresholdMethod;
            out.GradientThreshold = this.GradientThreshold;
            out.MaxEpochs = this.MaxEpochs;
            out.MiniBatchSize = this.MiniBatchSize;
            out.Verbose = this.Verbose;
            out.VerboseFrequency = this.VerboseFrequency;
            out.ValidationData = this.ValidationData;
            out.ValidationFrequency = this.ValidationFrequency;
            out.ValidationPatience = this.ValidationPatience;
            out.Shuffle = this.Shuffle;
            out.CheckpointPath = this.CheckpointPath;
            out.ExecutionEnvironment = this.ExecutionEnvironment;
            out.WorkerLoad = this.WorkerLoad;
            out.OutputFcn = this.OutputFcn;
            out.Plots = this.Plots;
            out.SequenceLength = this.SequenceLength;
            out.SequencePaddingValue = this.SequencePaddingValue;
            out.isGuided = this.isGuided;
            out.Rho = this.Rho;
            out.RevisitBatchNum = this.RevisitBatchNum;
            out.VerificationSetNum = this.VerificationSetNum;
        end
    end
    
    methods(Static)
        function [inputArguments,extraArgs] = parseInputArguments(varargin)
            try
                parser = iCreateParser();
                parser.parse(varargin{:});
                inputArguments = iConvertToCanonicalForm(parser);
                extraArgs = iConvertStructToArgList(parser.Unmatched);
            catch e
                % Reduce the stack trace of the error message by throwing as caller
                throwAsCaller(e)
            end
        end
        
        function out = flattenLearnRateScheduleSettings(out, learnRateScheduleSettings)
            learnRateScedule = learnRateScheduleSettings.Method;
            out.LearnRateSchedule = learnRateScedule;
            switch learnRateScedule
                case 'none'
                    out.LearnRateDropFactor = [];
                    out.LearnRateDropPeriod = [];
                case 'piecewise'
                    out.LearnRateDropFactor = learnRateScheduleSettings.DropRateFactor;
                    out.LearnRateDropPeriod = learnRateScheduleSettings.DropPeriod;
                otherwise
                    error(message('nnet_cnn:TrainingOptionsSGDM:InvalidLearningRateScheduleMethod'));
            end
        end
        
        function errorForInvalidOptions(s,solverName)
            fieldNames = fieldnames(s);
            numFields = numel(fieldNames);
            if ( numFields > 0 )
                firstInvalidFieldName = fieldNames{1};
                error(message('nnet_cnn:TrainingOptionsSGDM:InvalidTrainingOptionForSolver',firstInvalidFieldName,solverName));
            end
        end
    end
end

function argList = iConvertStructToArgList(s)
fieldNames = fieldnames(s);
numFields = numel(fieldNames);
argList = cell(1,2*numFields);
for i = 1:numFields
    argList{2*i-1} = fieldNames{i};
    argList{2*i} = s.(fieldNames{i});
end
end

function p = iCreateParser()
p = inputParser;
p.KeepUnmatched = true;

defaultLearnRateSchedule = 'none';
defaultLearnRateDropFactor = 0.1;
defaultLearnRateDropPeriod = 10;
defaultL2Regularization = 0.0001;
defaultGradientThresholdMethod = 'l2norm';
defaultGradientThreshold = Inf;
defaultMaxEpochs = 30;
defaultMiniBatchSize = 128;
defaultVerbose = true;
defaultVerboseFrequency = 50;
defaultValidationData = [];
defaultValidationFrequency = 50;
defaultValidationPatience = 5;
defaultShuffle = 'once';
defaultCheckpointPath = '';
defaultExecutionEnvironment = 'auto';
defaultWorkerLoad = [];
defaultOutputFcn = [];
defaultPlots = 'none';
defaultSequenceLength = 'longest';
defaultSequencePaddingValue = 0;
defaultisGuided = false;
defaultRho = 4;
defaultRevisitBatchNum = 2;
defaultVerificationSetNum = 4;

p.addParameter('LearnRateSchedule', defaultLearnRateSchedule, @(x)any(iAssertAndReturnValidLearnRateSchedule(x)));
p.addParameter('LearnRateDropFactor', defaultLearnRateDropFactor, @iAssertValidLearnRateDropFactor);
p.addParameter('LearnRateDropPeriod', defaultLearnRateDropPeriod, @iAssertIsPositiveIntegerScalar);
p.addParameter('L2Regularization', defaultL2Regularization, @iAssertValidL2Regularization);
p.addParameter('GradientThresholdMethod', defaultGradientThresholdMethod, @(x)any(iAssertandReturnValidGradientThresholdMethod(x)));
p.addParameter('GradientThreshold', defaultGradientThreshold, @iAssertValidGradientThreshold);
p.addParameter('MaxEpochs', defaultMaxEpochs, @iAssertIsPositiveIntegerScalar);
p.addParameter('MiniBatchSize', defaultMiniBatchSize, @iAssertIsPositiveIntegerScalar);
p.addParameter('Verbose', defaultVerbose, @iAssertValidVerbose);
p.addParameter('VerboseFrequency', defaultVerboseFrequency, @iAssertIsPositiveIntegerScalar);
p.addParameter('ValidationData', defaultValidationData, @iAssertValidValidationData);
p.addParameter('ValidationFrequency', defaultValidationFrequency, @iAssertIsPositiveIntegerScalar);
p.addParameter('ValidationPatience', defaultValidationPatience, @iAssertValidValidationPatience);
p.addParameter('Shuffle', defaultShuffle, @(x)any(iAssertAndReturnValidShuffleValue(x)));
p.addParameter('CheckpointPath', defaultCheckpointPath, @iAssertValidCheckpointPath);
p.addParameter('ExecutionEnvironment', defaultExecutionEnvironment, @(x)any(iAssertAndReturnValidExecutionEnvironment(x)));
p.addParameter('WorkerLoad', defaultWorkerLoad, @iAssertValidWorkerLoad);
p.addParameter('OutputFcn', defaultOutputFcn, @iAssertValidOutputFcn);
p.addParameter('Plots', defaultPlots, @iAssertValidPlots);
p.addParameter('SequenceLength', defaultSequenceLength, @(x)any(iAssertAndReturnValidSequenceLength(x)) );
p.addParameter('SequencePaddingValue', defaultSequencePaddingValue, @iAssertValidSequencePaddingValue);
p.addParameter('isGuided', defaultisGuided, @iAssertisGuided);
p.addParameter('Rho', defaultRho, @iAssertIsPositiveIntegerScalar);
p.addParameter('RevisitBatchNum', defaultRevisitBatchNum, @iAssertIsPositiveIntegerScalar);
p.addParameter('VerificationSetNum', defaultVerificationSetNum, @iAssertIsPositiveIntegerScalar);
end

function inputArguments = iConvertToCanonicalForm(parser)
results = parser.Results;
inputArguments = struct;
inputArguments.LearnRateSchedule = results.LearnRateSchedule;
inputArguments.LearnRateDropFactor = results.LearnRateDropFactor;
inputArguments.LearnRateDropPeriod = results.LearnRateDropPeriod;
inputArguments.L2Regularization = results.L2Regularization;
inputArguments.GradientThresholdMethod = iAssertandReturnValidGradientThresholdMethod(results.GradientThresholdMethod);
inputArguments.GradientThreshold = results.GradientThreshold;
inputArguments.MaxEpochs = results.MaxEpochs;
inputArguments.MiniBatchSize = results.MiniBatchSize;
inputArguments.Verbose = logical(results.Verbose);
inputArguments.VerboseFrequency = results.VerboseFrequency;
inputArguments.ValidationData = results.ValidationData;
inputArguments.ValidationFrequency = results.ValidationFrequency;
inputArguments.ValidationPatience = results.ValidationPatience;
inputArguments.Shuffle = iAssertAndReturnValidShuffleValue(results.Shuffle);
inputArguments.CheckpointPath = results.CheckpointPath;
inputArguments.ExecutionEnvironment = iAssertAndReturnValidExecutionEnvironment(results.ExecutionEnvironment);
inputArguments.WorkerLoad = results.WorkerLoad;
inputArguments.OutputFcn = results.OutputFcn;
inputArguments.Plots = iAssertAndReturnValidPlots(results.Plots);
inputArguments.SequenceLength = iAssertAndReturnValidSequenceLength(results.SequenceLength);
inputArguments.SequencePaddingValue = results.SequencePaddingValue;
inputArguments.isGuided = logical(results.isGuided);
inputArguments.Rho = results.Rho;
inputArguments.RevisitBatchNum = results.RevisitBatchNum;
inputArguments.VerificationSetNum = results.VerificationSetNum;
end

function scheduleSettings = iCreateLearnRateScheduleSettings( ...
    learnRateSchedule, learnRateDropFactor, learnRateDropPeriod)
scheduleSettings = struct;
learnRateSchedule = iAssertAndReturnValidLearnRateSchedule(learnRateSchedule);
switch learnRateSchedule
    case 'none'
        scheduleSettings.Method = 'none';
    case 'piecewise'
        scheduleSettings.Method = 'piecewise';
        scheduleSettings.DropRateFactor = learnRateDropFactor;
        scheduleSettings.DropPeriod = learnRateDropPeriod;
    otherwise
        error(message('nnet_cnn:TrainingOptionsSGDM:InvalidLearningRateScheduleMethod'));
end
end

function iAssertValidCheckpointPath(x)
% iAssertValidCheckpointPath Throws an error if the checkpoint path is not
% valid. Valid checkpoint paths are empty strings and existing directories
% with write access.
isEmptyPath = isempty(x);
isWritableExistingDir = ischar(x) && isdir(x) && iCanWriteToDir(x);
isValidCheckpointPath = isEmptyPath || isWritableExistingDir;

if ~isValidCheckpointPath
    iThrowCheckpointPathError()
end
end

function iAssertIsPositiveIntegerScalar(x)
validateattributes(x,{'numeric'}, ...
    {'scalar','integer','positive'});
end

function iAssertValidValidationPatience(x)
isValidPatience = isscalar(x) && (isinf(x) || isPositiveInteger(x));
if ~isValidPatience
    error(message('nnet_cnn:TrainingOptionsSGDM:InvalidValidationPatience'))
end
end

function tf = isPositiveInteger(x)
isPositive = x>0;
isInteger = isreal(x) && isnumeric(x) && all(mod(x,1)==0);
tf = isPositive && isInteger;
end

function iAssertValidL2Regularization(x)
validateattributes(x,{'numeric'}, ...
    {'scalar','real','finite','nonnegative'});
end

function method = iAssertandReturnValidGradientThresholdMethod(s)
method = validatestring(s, {'global-l2norm','l2norm','absolute-value'},...
    mfilename, 'GradientThresholdMethod');
end

function iAssertValidGradientThreshold(x)
validateattributes(x,{'numeric'}, ...
    {'scalar','real','nonnegative','nonempty','nonnan'});
end

function iAssertValidVerbose(x)
validateattributes(x,{'logical','numeric'}, ...
    {'scalar','binary'});
end

function iAssertisGuided(x)
validateattributes(x,{'logical','numeric'}, ...
    {'scalar','binary'});
end

function iAssertValidLearnRateDropFactor(x)
validateattributes(x,{'numeric'}, ...
    {'scalar','real','finite','>=',0,'<=',1});
end

function iAssertValidWorkerLoad(w)
if isempty(w)
    % an empty worker load value is valid
    return
end
validateattributes(w,{'numeric'}, ...
    {'vector','finite','nonnegative'});
if sum(w)<=0 || ( isscalar(w) && w > 1 && floor(w) ~= w )
    error(message('nnet_cnn:TrainingOptionsSGDM:InvalidWorkerLoad'));
end
end

function iAssertValidValidationData(validationData)
% iAssertValidValidationData   Return true if validationData is one of the
% allowed data types for validation. This can be either a table, an
% imageDatastore or a cell array containing two arrays. The consistency of
% the data with respect to training data and network architecture will be
% checked outside.
if istable(validationData)
    % data type is accepted, no further validation
elseif iIsAnImageDatastore(validationData)
    iAssertValidationDatastoreHasLabels(validationData);
elseif iIsAMiniBatchableDatastore(validationData)
    iAssertValidationMiniBatchableDatastoreHasLabels(validationData)
elseif iscell(validationData)
    iAssertValidationCellDataHasTwoEntries(validationData)
else
    iThrowValidationDataError();
end
end

function tf = iIsAnImageDatastore(x)
tf = isa(x, 'matlab.io.datastore.ImageDatastore');
end

function tf = iIsAMiniBatchableDatastore(x)
tf = isa(x,'matlab.io.Datastore') && isa(x,'matlab.io.datastore.MiniBatchable');
end

function iAssertValidationDatastoreHasLabels(imds)
if isempty(imds.Labels)
    error(message('nnet_cnn:TrainingOptionsSGDM:ImageDatastoreHasNoLabels'));
end
end

function iAssertValidationMiniBatchableDatastoreHasLabels(ds)

origMiniBatchSize = ds.MiniBatchSize;
ds.MiniBatchSize = 1;
hasResponses = size(preview(ds),2) > 1;
ds.MiniBatchSize = origMiniBatchSize;
if ~hasResponses
    error(message('nnet_cnn:TrainingOptionsSGDM:MiniBatchDatastoreHasNoResponses'));
end
end

function iAssertValidationCellDataHasTwoEntries(dataCell)
if numel(dataCell)~=2
    error(message('nnet_cnn:TrainingOptionsSGDM:CellArrayNeedsTwoEntries'));
end
end

function iThrowValidationDataError()
error(message('nnet_cnn:TrainingOptionsSGDM:InvalidValidationDataType'));
end

function shuffleValue = iAssertAndReturnValidShuffleValue(x)
expectedShuffleValues = {'never', 'once', 'every-epoch'};
shuffleValue = validatestring(x, expectedShuffleValues);
end

function learnRateScheduleValue = iAssertAndReturnValidLearnRateSchedule(x)
expectedLearnRateScheduleValues = {'none', 'piecewise'};
learnRateScheduleValue = validatestring(x, expectedLearnRateScheduleValues);
end

function validString = iAssertAndReturnValidExecutionEnvironment(inputString)
validExecutionEnvironments = {'auto', 'gpu', 'cpu', 'multi-gpu', 'parallel'};
validString = validatestring(inputString, validExecutionEnvironments);
end

function iAssertValidPlots(inputString)
validPlots = {'training-progress', 'none'};
validatestring(inputString, validPlots);
end

function validString = iAssertAndReturnValidPlots(inputString)
validPlots = {'training-progress', 'none'};
validString = validatestring(inputString, validPlots, {'char', 'string'});
validString = char(validString);
end

function y = iAssertAndReturnValidSequenceLength( x )
try
    if ischar(x) || isstring(x)
        y = validatestring(x, {'longest', 'shortest'});
    else
        validateattributes(x, {'numeric'}, {'scalar', 'real', 'integer', 'positive'});
        y = x;
    end
catch
    error(message('nnet_cnn:TrainingOptionsSGDM:InvalidSequenceLength'));
end
end

function iAssertValidSequencePaddingValue( x )
validateattributes(x, {'numeric'}, {'scalar', 'real'} )
end

function iThrowCheckpointPathError()
error(message('nnet_cnn:TrainingOptionsSGDM:InvalidCheckpointPath'));
end

function tf = iCanWriteToDir(proposedDir)
[~, status] = fileattrib(proposedDir);
tf = status.UserWrite;
end

function iAssertValidOutputFcn(f)
isValidFcn = isempty(f) || iIsFunctionWithInputs(f) || iIsCellOfValidFunctions(f);
if ~isValidFcn
    error(message('nnet_cnn:TrainingOptionsSGDM:InvalidOutputFcn'));
end
end

function tf = iIsFunctionWithInputs( f )
tf = isa(f, 'function_handle') && nargin(f) ~= 0;
end

function tf = iIsCellOfValidFunctions(f)
tf = iscell(f) && all( cellfun(@iIsFunctionWithInputs,f(:)) );
end