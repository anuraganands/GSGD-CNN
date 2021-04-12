classdef RegularizerFactory
    % RegularizerFactory   Class for creating regularizers.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods(Static)
        function regularizer = create(name,learnableParameters,precision,regularizationOptions)
            % regularizer = create(name,learnableParameters,precision,regularizationOptions)
            % creates a Regularizer object specified by the character array
            % name. Output regularizer specifies the type of regularization
            % to use when optimizing parameters in learnableParameters
            % using floating point precision specified in precision and
            % additional options specified in regularizationOptions. The
            % class of inputs is as follows:
            %
            %    learnableParameters   - an array of objects of type nnet.internal.cnn.layer.learnable.LearnableParameter
            %    precision             - an object of type nnet.internal.cnn.util.Precision
            %    regularizationOptions - an object of a subclass of nnet.cnn.TrainingOptions
            
            if iIsRegularizerL2(name)
                regularizer = nnet.internal.cnn.regularizer.RegularizerL2(learnableParameters,precision,regularizationOptions);
            else
                error('Unsupported regularizer.');
            end
        end
    end
end

function tf = iIsRegularizerL2(name)
tf = strcmpi(name,'l2');
end