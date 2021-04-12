classdef SolverSGDM < nnet.internal.cnn.solver.Solver
    % SolverSGDM   Stochastic gradient descent with momentum (SGDM) solver.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % Momentum   Momentum for SGDM update.
        %   A real scalar in [0,1] specifying the coefficient of the
        %   momentum term in SGDM update.
        Momentum
    end
    
    properties(Access=protected)
        % PreviousVelocities   Previous update steps in SGDM.
        %   A cell array of length NumLearnableParameters. Each element of
        %   the cell array contains the update step computed in the
        %   previous iteration of SGDM for that learnable parameter.
        PreviousVelocities
    end
    
    methods(Hidden)
        function this = SolverSGDM(learnableParameters,precision,options)
            % this = SolverSGDM(learnableParameters,precision,options)
            % creates a SolverSGDM object for optimizing parameters in
            % learnableParameters using floating point precision specified
            % in precision. The class of inputs is as follows:
            %
            %    learnableParameters - an array of objects of type nnet.internal.cnn.layer.learnable.LearnableParameter
            %    precision           - an object of type nnet.internal.cnn.util.Precision
            %    options             - an object of type nnet.cnn.TrainingOptionsSGDM
            
            this = this@nnet.internal.cnn.solver.Solver(learnableParameters,precision);
            this.Momentum = precision.cast(options.Momentum);
            initializeState(this);
        end
    end
    
    methods(Access=protected)
        function initializeState(this)
            % initializeState(this) sets the state of the solver to its
            % initial state.
            
            this.PreviousVelocities = iInitializeVelocities(this.NumLearnableParameters,this.Precision);
        end
    end
    
    methods
        function step = calculateUpdate(this,gradients,globalLearnRate)
            % step = calculateUpdate(this,gradients,globalLearnRate)
            % calculates the update for learnable parameters by applying
            % one step of SGDM solver. Input gradients is a cell array
            % where each element is the total gradient of the objective
            % (loss + regularization) with respect to one learnable
            % parameter. Input globalLearnRate specifies the global
            % learning rate to use for calculating the update step. The
            % length of gradients must equal NumLearnableParameters.
            
            localLearnRates = this.LocalLearnRates;
            numLearnableParameters = this.NumLearnableParameters;
            momentum = this.Momentum;
            
            step = cell(1,numLearnableParameters);
            for i = 1:numLearnableParameters
                % No update needed for parameters that are not learning
                if localLearnRates{i} ~= 0 && ~isempty(gradients{i})
                    effectiveLearningRate = globalLearnRate.*localLearnRates{i};
                    this.PreviousVelocities{i} = momentum.*this.PreviousVelocities{i} - effectiveLearningRate.*gradients{i};
                    step{i} = this.PreviousVelocities{i};
                end
            end
        end
    end
end

function velocities = iInitializeVelocities(numLearnableParameters,precision)
velocities = cell(1,numLearnableParameters);
for i = 1:numLearnableParameters
    velocities{i} = precision.zeros(1);
end
end