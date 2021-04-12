classdef(Abstract) Regularizer
    % Regularizer   Abstract class for defining new regularizers.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % Precision   A nnet.internal.cnn.util.Precision object.
        %   A nnet.internal.cnn.util.Precision object specifying the
        %   precision to use for floating point calculations.
        Precision
    end
    
    methods(Access=protected)
        function this = Regularizer(precision)
            % this = Regularizer(precision) creates a Regularizer object
            % that specifies the type of regularization for the loss
            % function that is being optimized.
            %
            %    precision - an object of type nnet.internal.cnn.util.Precision
            
            this.Precision = precision;
        end
    end
    
    methods(Abstract)
        regularizedLoss = regularizeLoss(this,loss,learnableParameters);
        regularizedGradients = regularizeGradients(this,gradients,learnableParameters);
    end
end