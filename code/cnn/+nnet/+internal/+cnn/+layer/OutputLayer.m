classdef (Abstract) OutputLayer < nnet.internal.cnn.layer.Layer
    % OutputLayer     Interface for convolutional neural network output layers
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % InputNames   Output layers have one input
        InputNames = {'in'}
        
        % OutputNames   Output layers have no outputs
        OutputNames = {}
    end
    
    methods (Abstract)
        % forwardLoss    Return the loss between the output obtained from
        % the network and the expected output
        %
        % Inputs
        %   anOutputLayer - the output layer to forward the loss thru
        %   Z - the output from forward propagation thru the layer
        %   T - the expected output
        %
        % Outputs
        %   loss - the loss between Z and T
        loss = forwardLoss( anOutputLayer, Z, T)
        
        % backwardLoss    Back propagate the derivative of the loss function
        %
        % Inputs
        %   anOutputLayer - the output layer to backprop the loss thru
        %   Z - the output from forward propagation thru the layer
        %   T - the expected output
        %
        % Outputs
        %   dX - the derivative of the loss function with respect to X
        dX = backwardLoss( anOutputLayer, Z, T)
    end
    
    methods (Sealed)
        function Z = predict( ~, X )
            % predict   Forward input data through the layer and output the
            % result
            Z = X;
        end
        
        function [Z, memory] = forward( ~, X )
            % forward   Forward input data through the layer and output the
            % result
            Z = X;
            memory = [];
        end
        
        function [dX,dW] = backward( ~, ~, ~, ~, ~ ) %#ok<STOUT>
            
            % Throw an error since it should not be possible to call
            % backward on this layer
            iThrow('nnet_cnn:internal:cnn:layer:OutputLayer:BackwardProhibited');
        end
    end
end

function iThrow( msg )
error( message( msg ) )
end