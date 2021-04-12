classdef RegularizerL2 < nnet.internal.cnn.regularizer.Regularizer
    % RegularizerL2   Abstract class for representing L2 regularization.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % NumLearnableParameters   Number of learnable parameters.
        %   A non-negative integer specifying the number of learnable
        %   parameters in the optimization problem.
        NumLearnableParameters
        
        % LocalL2Factors   Local L2 factors for learnable parameters.
        %   A cell array of length NumLearnableParameters where each
        %   element of the cell array is a non-negative scalar that
        %   specifies the local L2 factor for that learnable parameter.
        %   Every learnable parameter can define its own local L2 factor.
        %   The effective L2 factor for each learnable parameter is
        %   calculated by taking the product of the global L2 factor and
        %   its local L2 factor.
        LocalL2Factors
        
        % GlobalL2Factor   Global L2 factor for learnable parameters.
        %   A non-negative scalar specifying the global L2 factor for
        %   learnable parameters.
        GlobalL2Factor
    end
    
    methods(Hidden)
        function this = RegularizerL2(learnableParameters,precision,options)
            % this = RegularizerL2(learnableParameters,precision,options)
            % creates a RegularizerL2 object representing L2 regularization
            % for the loss function that is being optimized using floating
            % point precision specified in precision. The class of inputs
            % is as follows:
            %
            %    learnableParameters - an array of objects of type nnet.internal.cnn.layer.learnable.LearnableParameter
            %    precision           - an object of type nnet.internal.cnn.util.Precision
            %    options             - an object of a subclass of nnet.cnn.TrainingOptions
            
            this = this@nnet.internal.cnn.regularizer.Regularizer(precision);
            this.NumLearnableParameters = numel(learnableParameters);
            this.LocalL2Factors = iExtractLocalL2Factors(learnableParameters,precision);
            this.GlobalL2Factor = precision.cast(options.L2Regularization);
        end
    end
    
    methods
        function regularizedLoss = regularizeLoss(this,loss,learnableParameters)
            % regularizedLoss = regularizeLoss(this,loss,learnableParameters)
            % takes a scalar loss representing the loss function value,
            % adds the contribution of the regularization term and returns
            % the updated objective (loss + regularization). Input
            % learnableParameters is an array of objects of type
            % nnet.internal.cnn.layer.learnable.LearnableParameter.
            
            numLearnableParameters = this.NumLearnableParameters;
            globalL2Factor = this.GlobalL2Factor;
            regularizedLoss = loss;
            
            for i = 1:numLearnableParameters
                effectiveL2Factor = this.LocalL2Factors{i}*globalL2Factor;
                weights = learnableParameters(i).Value;
                regularizedLoss = regularizedLoss + 0.5*effectiveL2Factor*sum(weights(:).^2);
            end
        end
        
        function regularizedGradients = regularizeGradients(this,gradients,learnableParameters)
            % regularizedGradients = regularizeGradients(this,gradients,learnableParameters)
            % takes a cell array gradients where each element is the
            % gradient of the loss function with respect to one learnable
            % parameter, adds the contribution of the regularization term
            % and returns the updated gradients. Output
            % regularizedGradients is a cell array of the same length as
            % gradients. Each element of regularizedGradients is the
            % gradient of the objective function (loss + regularization)
            % with respect to one learnable parameter. Input
            % learnableParameters is an array of objects of type
            % nnet.internal.cnn.layer.learnable.LearnableParameter. The
            % length of gradients and learnableParameters must equal
            % NumLearnableParameters.
            
            numLearnableParameters = this.NumLearnableParameters;
            globalL2Factor = this.GlobalL2Factor;
            regularizedGradients = gradients;
            
            for i = 1:numLearnableParameters
                effectiveL2Factor = this.LocalL2Factors{i}*globalL2Factor;
                % We don't need to calculate the regularized gradient for
                % parameters that are not learning since it will go unused
                if learnableParameters(i).LearnRateFactor ~= 0 && ~isempty(regularizedGradients{i})
                    weights = learnableParameters(i).Value;
                    regularizedGradients{i} = effectiveL2Factor.*weights + regularizedGradients{i};
                end
            end
        end
    end
end

function localL2Factors = iExtractLocalL2Factors(learnableParameters,precision)
numLearnableParameters = numel(learnableParameters);
localL2Factors = cell(1,numLearnableParameters);
for i = 1:numLearnableParameters
    localL2Factors{i} = precision.cast(learnableParameters(i).L2Factor);
end
end