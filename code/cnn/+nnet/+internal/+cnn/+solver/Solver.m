classdef(Abstract) Solver < handle
    % Solver   Abstract class for defining new solvers.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % NumLearnableParameters   Number of learnable parameters.
        %   A non-negative integer specifying the number of learnable
        %   parameters in the optimization problem.
        NumLearnableParameters
        
        % LocalLearnRates   Local learning rates for learnable parameters.
        %   A cell array of length NumLearnableParameters where each
        %   element of the cell array specifies the local learning rate for
        %   that learnable parameter. Every learnable parameter can define
        %   its own local learning rate. The effective learning rate for
        %   each learnable parameter is calculated by taking the product of
        %   the current global learning rate and its local learning rate.
        LocalLearnRates
        
        % Precision   A nnet.internal.cnn.util.Precision object.
        %   A nnet.internal.cnn.util.Precision object specifying the
        %   precision to use for floating point calculations.
        Precision
    end
    
    methods(Access=protected)
        function this = Solver(learnableParameters,precision)
            % this = Solver(learnableParameters,precision) creates a Solver
            % object for optimizing parameters in learnableParameters using
            % floating point precision specified in precision. The class of
            % inputs is as follows:
            %
            %    learnableParameters - an array of objects of type nnet.internal.cnn.layer.learnable.LearnableParameter
            %    precision           - an object of type nnet.internal.cnn.util.Precision
            
            this.NumLearnableParameters = numel(learnableParameters);
            this.Precision = precision;
            this.LocalLearnRates = iExtractLocalLearnRates(learnableParameters,precision);
        end
    end
    
    methods(Abstract,Access=protected)
        initializeState(this);
    end
    
    methods(Abstract)
        step = calculateUpdate(this,gradients,globalLearnRate);
    end
end

function localLearnRates = iExtractLocalLearnRates(learnableParameters,precision)
numLearnableParameters = numel(learnableParameters);
localLearnRates = cell(1,numLearnableParameters);
for i = 1:numLearnableParameters
    localLearnRates{i} = precision.cast(learnableParameters(i).LearnRateFactor);
end
end