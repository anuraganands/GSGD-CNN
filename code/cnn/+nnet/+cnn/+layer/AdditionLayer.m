classdef AdditionLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % AdditionLayer   Addition layer
    %
    %   To create a addition layer layer, use additionLayer.
    %
    %   This layer takes multiple inputs with the same height and width and
    %   adds them together.
    %
    %   AdditionLayer properties:
    %       Name                        - A name for the layer.
    %       NumInputs                   - The number of inputs for the
    %                                     layer.
    %
    %   Example:
    %       Create a addition layer layer.
    %
    %       layer = additionLayer(3);
    %
    %   See also convolution2dLayer
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Dependent)
        % Name   A name for the layer
        Name
    end
    
    properties (SetAccess=private, Dependent)
        % NumInputs   Number of inputs for this layer
        NumInputs
    end
    
    methods
        function this = AdditionLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
        
        function val = get.NumInputs(this)
            val = this.PrivateLayer.NumInputs;
        end
        
        function out = saveobj(this)
            out.Version = 1.0;
            out.Name = this.PrivateLayer.Name;
            out.NumInputs = this.PrivateLayer.NumInputs;
        end
    end
    
    methods(Static)
        function this = loadobj(in)
            internalLayer = nnet.internal.cnn.layer.Addition(in.Name, in.NumInputs);
            this = nnet.cnn.layer.AdditionLayer(internalLayer);
        end
    end
    
    methods(Access = protected)
        function [description, type] = getOneLineDisplay(this)
            description = iGetMessageString('nnet_cnn:layer:AdditionLayer:oneLineDisplay', this.NumInputs);
            
            type = iGetMessageString( 'nnet_cnn:layer:AdditionLayer:Type' );
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end