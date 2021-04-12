classdef TrainingOptionsRMSProp < nnet.cnn.TrainingOptions
    % TrainingOptionsRMSProp   Training options for root mean square propagation (RMSProp)
    %
    %   This class holds the training options for root mean square
    %   propagation (RMSProp)
    %
    %   TrainingOptionsRMSProp properties:
    %       SquaredGradientDecayFactor  - Decay factor for moving average
    %                                     of squared gradients.
    %       Epsilon                     - Offset for the denominator in the
    %                                     RMSProp update.
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
    %
    %   Example:
    %       Create a set of training options for training with RMSProp. The
    %       learning rate will be reduced by a factor of 0.2 every 5
    %       epochs. The training will last for 20 epochs, and each
    %       iteration will use a mini-batch with 300 observations.
    %
    %       opts = trainingOptions('rmsprop', ...
    %           'Plots', 'training-progress', ...
    %           'LearnRateSchedule', 'piecewise', ...
    %           'LearnRateDropFactor', 0.2, ...
    %           'LearnRateDropPeriod', 5, ...
    %           'MaxEpochs', 20, ...
    %           'MiniBatchSize', 300);
    %
    %   See also trainingOptions, trainNetwork.
    
    % Copyright 2017 The MathWorks, Inc.
    
    properties(Access = protected)
        % Version   Number to identify the current version of this object
        %   This is used to ensure that objects from older versions are
        %   loaded correctly.
        Version = 1
    end
    
    properties(SetAccess = private)
        % SquaredGradientDecayFactor   Decay factor for moving average of squared gradients
        %   A real scalar in [0,1) specifying the exponential decay rate
        %   for the squared gradient moving average.
        SquaredGradientDecayFactor
        
        % Epsilon   Offset for the denominator in the RMSProp update
        %   A positive real scalar specifying the offset to use in the
        %   denominator for the RMSProp update to prevent divide-by-zero
        %   problems.
        Epsilon
        
        % InitialLearnRate   Initial learning rate
        %   The initial learning rate that is used for training. If the
        %   learning rate is too low, training will take a long time, but
        %   if it is too high, the training is likely to get stuck at a
        %   suboptimal result.
        InitialLearnRate
    end
    
    methods(Access = public)
        function this = TrainingOptionsRMSProp(inputArguments)
            this = this@nnet.cnn.TrainingOptions(inputArguments);
            this.SquaredGradientDecayFactor = inputArguments.SquaredGradientDecayFactor;
            this.Epsilon = inputArguments.Epsilon;
            this.InitialLearnRate = inputArguments.InitialLearnRate;
        end
        
        function out = saveobj(this)
            out = saveobj@nnet.cnn.TrainingOptions(this);
            out.SquaredGradientDecayFactor = this.SquaredGradientDecayFactor;
            out.Epsilon = this.Epsilon;
            out.InitialLearnRate = this.InitialLearnRate;
        end
    end
    
    methods(Static)
        function inputArguments = parseInputArguments(varargin)
            try
                [inputArguments,extraArgs] = nnet.cnn.TrainingOptions.parseInputArguments(varargin{:});
                parser = iCreateParser();
                parser.parse(extraArgs{:});
                nnet.cnn.TrainingOptions.errorForInvalidOptions(parser.Unmatched,'rmsprop');
                inputArguments = iConvertToCanonicalForm(parser,inputArguments);
            catch e
                % Reduce the stack trace of the error message by throwing as caller
                throwAsCaller(e)
            end
        end
        
        function this = loadobj(in)
            this = nnet.cnn.TrainingOptionsRMSProp(in);
        end
    end
end

function p = iCreateParser()
p = inputParser;
p.KeepUnmatched = true;

defaultSquaredGradientDecayFactor = 0.9;
defaultEpsilon = 1e-8;
defaultInitialLearnRate = 0.001;

p.addParameter('SquaredGradientDecayFactor', defaultSquaredGradientDecayFactor, @iAssertValidSquaredGradientDecayFactor);
p.addParameter('Epsilon', defaultEpsilon, @iAssertValidEpsilon);
p.addParameter('InitialLearnRate', defaultInitialLearnRate, @iAssertValidInitialLearnRate);
end

function inputArguments = iConvertToCanonicalForm(parser,inputArguments)
results = parser.Results;
inputArguments.SquaredGradientDecayFactor = results.SquaredGradientDecayFactor;
inputArguments.Epsilon = results.Epsilon;
inputArguments.InitialLearnRate = results.InitialLearnRate;
end

function iAssertValidSquaredGradientDecayFactor(x)
validateattributes(x,{'numeric'}, ...
    {'scalar','real','finite','>=',0,'<',1});
end

function iAssertValidEpsilon(x)
validateattributes(x,{'numeric'}, ...
    {'scalar','real','finite','>',0});
end

function iAssertValidInitialLearnRate(x)
validateattributes(x, {'numeric'}, ...
    {'scalar','real','finite','positive'});
end