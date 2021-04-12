classdef SolverADAM < nnet.internal.cnn.solver.Solver
    % SolverADAM   Adaptive Moment estimation (ADAM) solver.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % GradientDecayFactor   Decay factor for moving average of gradients.
        %   A real scalar in [0,1) specifying the exponential decay rate
        %   for the gradient moving average. This parameter is denoted by
        %   the symbol Beta1 in the ADAM paper.
        GradientDecayFactor
        
        % SquaredGradientDecayFactor   Decay factor for moving average of squared gradients.
        %   A real scalar in [0,1) specifying the exponential decay rate
        %   for the squared gradient moving average. This parameter is
        %   denoted by the symbol Beta2 in the ADAM paper.
        SquaredGradientDecayFactor
        
        % Epsilon   Offset for the denominator in the ADAM update.
        %   A positive real scalar specifying the offset to use in the
        %   denominator for the ADAM update to prevent divide-by-zero
        %   problems.
        Epsilon
    end
    
    properties(Access=protected)
        % GradientMovingAverage   Moving average of gradients.
        %   A cell array of length NumLearnableParameters. Each element of
        %   the cell array contains the moving average of gradient for that
        %   learnable parameter.
        GradientMovingAverage
        
        % SquaredGradientMovingAverage   Moving average of squared gradients.
        %   A cell array of length NumLearnableParameters. Each element of
        %   the cell array contains the moving average of the squared
        %   gradient for that learnable parameter.
        SquaredGradientMovingAverage
        
        % NumUpdates   Number of updates so far.
        %   A non-negative integer indicating the number of update steps
        %   that have been computed so far.
        NumUpdates
    end
    
    methods(Hidden)
        function this = SolverADAM(learnableParameters,precision,options)
            % this = SolverADAM(learnableParameters,precision,options)
            % creates a SolverADAM object for optimizing parameters in
            % learnableParameters using floating point precision specified
            % in precision. The class of inputs is as follows:
            %
            %    learnableParameters - an array of objects of type nnet.internal.cnn.layer.learnable.LearnableParameter
            %    precision           - an object of type nnet.internal.cnn.util.Precision
            %    options             - an object of type nnet.cnn.TrainingOptionsADAM
            
            this = this@nnet.internal.cnn.solver.Solver(learnableParameters,precision);
            this.GradientDecayFactor = precision.cast(options.GradientDecayFactor);
            this.SquaredGradientDecayFactor = precision.cast(options.SquaredGradientDecayFactor);
            this.Epsilon = precision.cast(options.Epsilon);
            initializeState(this);
        end
    end
    
    methods(Access=protected)
        function initializeState(this)
            % initializeState(this) sets the state of the solver to its
            % initial state.
            
            this.GradientMovingAverage = iInitializeMovingAverage(this.NumLearnableParameters,this.Precision);
            this.SquaredGradientMovingAverage = iInitializeMovingAverage(this.NumLearnableParameters,this.Precision);
            this.NumUpdates = 0;
        end
    end
    
    methods
        function step = calculateUpdate(this,gradients,globalLearnRate)
            % step = calculateUpdate(this,gradients,globalLearnRate)
            % calculates the update for learnable parameters by applying
            % one step of ADAM solver. Input gradients is a cell array
            % where each element is the total gradient of the objective
            % (loss + regularization) with respect to one learnable
            % parameter. Input globalLearnRate specifies the global
            % learning rate to use for calculating the update step. The
            % length of gradients must equal NumLearnableParameters.
            
            localLearnRates = this.LocalLearnRates;
            numLearnableParameters = this.NumLearnableParameters;
            
            beta1 = this.GradientDecayFactor;
            beta2 = this.SquaredGradientDecayFactor;
            epsilon = this.Epsilon;
            
            this.NumUpdates = this.NumUpdates + 1;
            learnRateShrinkFactor = iCalculateLearnRateShrinkFactor(beta1,beta2,this.NumUpdates);
            
            step = cell(1,numLearnableParameters);
            for i = 1:numLearnableParameters
                % No update needed for parameters that are not learning
                if localLearnRates{i} ~= 0 && ~isempty(gradients{i})
                    effectiveLearningRate = learnRateShrinkFactor.*globalLearnRate.*localLearnRates{i};
                    this.GradientMovingAverage{i} = beta1.*this.GradientMovingAverage{i} + (1 - beta1).*gradients{i};
                    this.SquaredGradientMovingAverage{i} = beta2.*this.SquaredGradientMovingAverage{i} + (1 - beta2).*(gradients{i}.^2);
                    step{i} = -effectiveLearningRate.*( this.GradientMovingAverage{i}./(sqrt(this.SquaredGradientMovingAverage{i}) + epsilon) );
                end
            end
        end
    end
end

function movingAverage = iInitializeMovingAverage(numLearnableParameters,precision)
movingAverage = cell(1,numLearnableParameters);
for i = 1:numLearnableParameters
    movingAverage{i} = precision.zeros(1);
end
end

function learnRateShrinkFactor = iCalculateLearnRateShrinkFactor(beta1,beta2,t)
learnRateShrinkFactor = sqrt(1-beta2^t)/(1-beta1^t);
end