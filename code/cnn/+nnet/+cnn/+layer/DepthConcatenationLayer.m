classdef DepthConcatenationLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % DepthConcatenationLayer   Depth concatenation layer
    %
    %   To create a depth concatenation layer, use depthConcatenationLayer.
    %
    %   This layer takes multiple inputs with the same height and width and
    %   concatenates them along the third dimension (channels).
    %
    %   DepthConcatenationLayer properties:
    %       Name                        - A name for the layer.
    %       NumInputs                   - The number of inputs for the
    %                                     layer.
    %
    %   Example:
    %       Create a depth concatenation layer with two inputs
    %
    %       layer = depthConcatenationLayer(2);
    %
    %   See also depthConcatenationLayer
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        Name
    end
    
    properties (SetAccess=private, Dependent)
        % NumInputs   Number of inputs for this layer
        NumInputs
    end
    
    methods
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
            concatenationAxis = 3;
            internalLayer = nnet.internal.cnn.layer.Concatenation(in.Name, concatenationAxis, in.NumInputs);
            this = nnet.cnn.layer.DepthConcatenationLayer(internalLayer);
        end
    end
    
    methods(Access = public)
        function this = DepthConcatenationLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
    end
    
    methods(Access = protected)
        function [description, type] = getOneLineDisplay(this)
            description = iGetMessageString('nnet_cnn:layer:DepthConcatenationLayer:oneLineDisplay', this.NumInputs);
            
            type = iGetMessageString( 'nnet_cnn:layer:DepthConcatenationLayer:Type' );
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end