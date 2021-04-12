classdef DropoutLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % DropoutLayer   Dropout layer
    %
    %   To create a dropout layer, use dropoutLayer
    %
    %   A layer which randomly drops neurons during training. This can be
    %   useful to prevent over-fitting.
    %
    %   DropoutLayer properties:
    %       Probability                 - The probability that a neuron
    %                                     will be dropped at training time.
    %       Name                        - A name for the layer.
    %
    %   Example:
    %       Create a dropout layer which will dropout roughly 40% of the input
    %       elements.
    %
    %       layer = dropoutLayer(0.4);
    %
    %   See also dropoutLayer
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
    
    properties(SetAccess = private, Dependent)
        % Probability   The probability that a neuron will be dropped.
        %   A number between 0 and 1 which is the probability that a neuron
        %   will be dropped during training. A higher number will result in
        %   more neurons being dropped during training.
        Probability
    end
    
    methods
        function this = DropoutLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 1.0;
            out.Name = privateLayer.Name;
            out.Probability = privateLayer.Probability;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
        
        function val = get.Probability(this)
            val = this.PrivateLayer.Probability;
        end
    end
    
    methods(Hidden, Static)
        function this = loadobj(in)
            internalLayer = nnet.internal.cnn.layer.Dropout(in.Name, in.Probability);
            this = nnet.cnn.layer.DropoutLayer(internalLayer);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            probabilityString = int2str( this.Probability*100 );
            description = iGetMessageString( ...
                'nnet_cnn:layer:DropoutLayer:oneLineDisplay', ...
                probabilityString );
            
            type = iGetMessageString( 'nnet_cnn:layer:DropoutLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            hyperparameters = {
                'Probability'
                };
            
            groups = [
                this.propertyGroupGeneral( {'Name'} )
                this.propertyGroupHyperparameters( hyperparameters )
                ];
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end