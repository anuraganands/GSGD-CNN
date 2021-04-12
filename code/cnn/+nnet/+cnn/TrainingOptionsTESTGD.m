classdef TrainingOptionsTESTGD < nnet.cnn.TrainingOptions
    % TrainingOptionsTESTGD   Training options for stochastic gradient descent with momentum
    %
    %   This class holds the training options for stochastic gradient
    %   descent with momentum.
    %
    %   TrainingOptionsTESTGD properties:
    %       Momentum                    - Momentum for learning.
    %       InitialLearnRate            - Initial learning rate.
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
    %
    %   Example:
    %       Create a set of training options for training with stochastic
    %       gradient descent with momentum. The learning rate will be
    %       reduced by a factor of 0.2 every 5 epochs. The training will
    %       last for 20 epochs, and each iteration will use a mini-batch
    %       with 300 observations.
    %
    %       opts = trainingOptions('sgdm', ...
    %           'Plots', 'training-progress', ...
    %           'LearnRateSchedule', 'piecewise', ...
    %           'LearnRateDropFactor', 0.2, ...
    %           'LearnRateDropPeriod', 5, ...
    %           'MaxEpochs', 20, ...
    %           'MiniBatchSize', 300);
    %
    %   See also trainingOptions, trainNetwork.
    
    % Copyright 2015-2017 The MathWorks, Inc.
    
    properties(Access = protected)
        % Version   Number to identify the current version of this object
        %   This is used to ensure that objects from older versions are
        %   loaded correctly.
        Version = 3
    end
    
    properties(SetAccess = private)
        % Momentum   Momentum for learning
        %   The momentum determines the contribution of the gradient step
        %   from the previous iteration to the current iteration of
        %   training. It must be a value between 0 and 1, where 0 will give
        %   no contribution from the previous step, and 1 will give a
        %   maximal contribution from the previous step.
        Momentum
        
        % InitialLearnRate   Initial learning rate
        %   The initial learning rate that is used for training. If the
        %   learning rate is too low, training will take a long time, but
        %   if it is too high, the training is likely to get stuck at a
        %   suboptimal result.
        InitialLearnRate
    end
    
    methods(Access = public)
        function this = TrainingOptionsTESTGD(inputArguments)
            this = this@nnet.cnn.TrainingOptions(inputArguments);
            this.Momentum = inputArguments.Momentum;
            this.InitialLearnRate = inputArguments.InitialLearnRate;
        end
        
        function out = saveobj(this)
            out = saveobj@nnet.cnn.TrainingOptions(this);
            out.Momentum = this.Momentum;
            out.InitialLearnRate = this.InitialLearnRate;
        end
    end
    
    methods(Static)
        function inputArguments = parseInputArguments(varargin)
            try
                [inputArguments,extraArgs] = nnet.cnn.TrainingOptions.parseInputArguments(varargin{:});
                parser = iCreateParser();
                parser.parse(extraArgs{:});
                nnet.cnn.TrainingOptions.errorForInvalidOptions(parser.Unmatched,'tgd');
                inputArguments = iConvertToCanonicalForm(parser,inputArguments);
            catch e
                % Reduce the stack trace of the error message by throwing as caller
                throwAsCaller(e)
            end
        end
        
        function this = loadobj(in)
            if iTrainingOptionsAreFrom2016aOr2016b(in)
                in = iUpgradeTrainingOptionsFrom2016aOr2016bTo2017a(in);
            end
            if iTrainingOptionsAreFrom2017a(in)
                in = iUpgradeTrainingOptionsFrom2017aTo2017b(in);
            end
            if iTrainingOptionsAreFrom2017b(in)
                in = iUpgradeTrainingOptionsFrom2017bTo2018a(in);
            end
            this = nnet.cnn.TrainingOptionsTESTGD(in);
        end
    end
end

function p = iCreateParser()
p = inputParser;
p.KeepUnmatched = true;

defaultMomentum = 0.9;
defaultInitialLearnRate = 0.01;

p.addParameter('Momentum', defaultMomentum, @iAssertValidMomentum);
p.addParameter('InitialLearnRate', defaultInitialLearnRate, @iAssertValidInitialLearnRate);
end

function inputArguments = iConvertToCanonicalForm(parser,inputArguments)
results = parser.Results;
inputArguments.Momentum = results.Momentum;
inputArguments.InitialLearnRate = results.InitialLearnRate;
end

function tf = iTrainingOptionsAreFrom2016aOr2016b(in)
% For training options from 2016a and 2016b, "in" will be an object
% instead of a struct.
tf = isa(in, 'nnet.cnn.TrainingOptionsTESTGD');
end

function tf = iTrainingOptionsAreFrom2017a(in)
% For training options from 2017a, Version will be 1
tf = in.Version == 1;
end

function tf = iTrainingOptionsAreFrom2017b(in)
% For training options from 2017b, Version will be 2
tf = in.Version == 2;
end

function inStruct = iUpgradeTrainingOptionsFrom2016aOr2016bTo2017a(in)
% iUpgradeTrainingOptionsFrom2016aOr2016bTo2017a   Upgrade training options
% from R2016a or R2016b to R2017a

% Set properties that exist in 2016a and 2016b
inStruct = struct;
inStruct.Momentum = in.Momentum;
inStruct.InitialLearnRate = in.InitialLearnRate;
inStruct = nnet.cnn.TrainingOptions.flattenLearnRateScheduleSettings(inStruct, in.LearnRateScheduleSettings);
inStruct.L2Regularization = in.L2Regularization;
inStruct.MaxEpochs = in.MaxEpochs;
inStruct.MiniBatchSize = in.MiniBatchSize;
inStruct.Verbose = in.Verbose;
inStruct.Shuffle = in.Shuffle;
inStruct.CheckpointPath = in.CheckpointPath;
inStruct.isGuided = in.isGuided;
inStruct.Rho = in.Rho;

% Set properties that don't exist in 2016a or 2016b
inStruct.VerboseFrequency = 50;
inStruct.ExecutionEnvironment = 'auto';
inStruct.WorkerLoad = [];
inStruct.OutputFcn = [];
inStruct.Version = 1;
end

function inStruct = iUpgradeTrainingOptionsFrom2017aTo2017b(inStruct)
% iUpgradeTrainingOptionsFrom2017aTo2017b   Upgrade training options
% from R2017a to R2017b

% Set properties that exist in 2017a
inStruct.Version = 2;

% Set properties that don't exist in 2017a
inStruct.ValidationData = [];
inStruct.ValidationFrequency = 50;
inStruct.ValidationPatience = 5;
inStruct.Plots = 'none';
inStruct.SequenceLength = 'longest';
inStruct.SequencePaddingValue = 0;
end

function inStruct = iUpgradeTrainingOptionsFrom2017bTo2018a(inStruct)
% iUpgradeTrainingOptionsFrom2017bTo2018a   Upgrade training options
% from R2017b to R2018a

% Set properties that exist in 2017b
inStruct.Version = 3;

% Set properties that don't exist in 2017b
inStruct.GradientThresholdMethod = 'l2norm';
inStruct.GradientThreshold = Inf;
end

function iAssertValidMomentum(x)
validateattributes(x,{'numeric'}, ...
    {'scalar','real','finite','>=',0,'<=',1});
end

function iAssertValidInitialLearnRate(x)
validateattributes(x, {'numeric'}, ...
    {'scalar','real','finite','positive'});
end
