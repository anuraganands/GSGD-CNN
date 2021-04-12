classdef VisualizationOutputLayer < nnet.internal.cnn.layer.Layer
    % VisualizationOutputLayer   Interface for convolutional neural network 
    % visualize output layers
    
    %   Copyright 2016 The MathWorks, Inc.

    properties(SetAccess = private)
        % InputNames   Output layers have one input
        InputNames = {'in'}
        
        % OutputNames   Output layers have no outputs
        OutputNames = {}
    end
    
    methods (Abstract)
        % forwardActivations    Return the activations of the layer
        %
        % Inputs
        %   anOutputLayer - the output layer to forward the loss thru
        %   X - the input to the layer
        %
        % Outputs
        %   activations - the activations
        activations = forwardActivations( anOutputLayer, X )
        
        % backwardActivations    Back propagate the derivative of the activations
        %
        % Inputs
        %   anOutputLayer - the output layer to backprop the activations thru
        %   X - the input to the layer
        %   Z - the output from forward propagation thru the layer
        %
        % Outputs
        %   dX - the derivative of the activations with respect to X
        dX = backwardActivations( anOutputLayer, X, Z )
    end
end
