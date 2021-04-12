classdef ReLULayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % ReLULayer   Rectified linear unit (ReLU) layer
    %
    %   To create a rectified linear unit layer, use reluLayer
    %
    %   A rectified linear unit layer. This type of layer performs a simple
    %   thresholding operation, where any input value less than zero will
    %   be set to zero.
    %
    %   ReLULayer properties:
    %       Name                        - A name for the layer.
    %
    %   Example:
    %       Create a relu layer.
    %
    %       layer = reluLayer()
    %
    %   See also reluLayer
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name         
    end
    
    methods
        function this = ReLULayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end      
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
        
        function out = saveobj(this)
            out.Version = 1.0;
            out.Name = this.PrivateLayer.Name;
        end
    end
    
    methods(Static)
        function this = loadobj(in)
            internalLayer = nnet.internal.cnn.layer.ReLU(in.Name);
            this = nnet.cnn.layer.ReLULayer(internalLayer);
        end
    end

    methods(Access = protected)
        function [description, type] = getOneLineDisplay(~)
            description = iGetMessageString('nnet_cnn:layer:ReLULayer:oneLineDisplay');
            
            type = iGetMessageString( 'nnet_cnn:layer:ReLULayer:Type' );
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end