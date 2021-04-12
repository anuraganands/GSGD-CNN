classdef (Abstract) ClassificationLayer < nnet.cnn.layer.Layer
    % ClassificationLayer   Interface for classification layers
    %
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties
        % Name (char vector)   A name for the layer
        Name = '';
        
        % ClassNames (cellstr)   The names of the classes
        %   A cell array containing the names of the classes. This will be
        %   automatically determined at training time. Prior to training,
        %   it will be empty.
        ClassNames
    end
    
    properties (SetAccess = protected)
        % Description (char vector)   A one line description for the layer
        Description
        
        % Type (char vector)   The type of layer
        Type
    end
    
    properties (Hidden)
        % NumClasses (scalar int)   Number of classes.
        %  This will be automatically determined at training time. Prior to
        %  training, it will be empty.
        NumClasses
    end
    
    methods (Abstract)
        % forwardLoss    Return the loss between the output obtained from
        % the network and the expected output
        %
        % Inputs
        %   this - the output layer to forward the loss through
        %   Y - Predictions made by network
        %   T - Targets (actual values)
        %
        % Outputs
        %   loss - the loss between Y and T
        loss = forwardLoss( this, Y, T)
        
        % backwardLoss    Back propagate the derivative of the loss function
        %
        % Inputs
        %   this - the output layer to backprop the loss through
        %   Y - Predictions made by network
        %   T - Targets (actual values)
        %
        % Outputs
        %   dLdY - the derivative of the loss (L) with respect to the input Y
        dLdY = backwardLoss( this, Y, T)
    end
    
    methods(Hidden, Access = protected)
        function layer = ClassificationLayer()
            try
                className = matlab.mixin.CustomDisplay.getClassNameForHeader( layer );
                nnet.internal.cnn.layer.util.CustomLayerVerifier...
                    .validateMethodSignatures(layer, className);
            catch e
                % Wrap exception in a CNNException, which reports the error in a custom way
                err = nnet.internal.cnn.util.CNNException.hBuildCustomError( e );
                throwAsCaller(err);
            end
        end

        function [description, type] = getOneLineDisplay( layer )
            if isempty( layer.Description )
                description = iGetMessageString( 'nnet_cnn:layer:ClassificationLayer:oneLineDisplay' );
            else
                description = layer.Description;
            end
            
            if isempty( layer.Type )
                type = iGetMessageString( 'nnet_cnn:layer:ClassificationLayer:Type' );
            else
                type = layer.Type;
            end
        end
    end
    
    methods
        function layer = set.Name( layer, val )
            iAssertValidLayerName( val );
            layer.Name = char( val );
        end
    end
end

function messageString = iGetMessageString( messageID )
messageString = getString( message( messageID ) );
end

function iAssertValidLayerName( name )
nnet.internal.cnn.layer.paramvalidation.validateLayerName( name );
end