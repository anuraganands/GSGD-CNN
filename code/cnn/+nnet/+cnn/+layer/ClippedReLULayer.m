classdef ClippedReLULayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % ClippedReLULayer   Clipped Rectified linear unit (ReLU) layer
    %
    %   To create a clipped rectified linear unit layer, use clippedReluLayer
    %
    %   A clipped rectified linear unit layer. This type of layer performs
    %   a simple threshold operation, where any input value less than zero
    %   will be set to zero and any value above the clipping ceiling will
    %   be set to the clipping ceiling. This is equivalent to:
    %     out = 0;         % For in<0
    %     out = in;        % For 0<=in<ceiling
    %     out = ceiling;   % For in>=ceiling
    %
    %   ClippedReLULayer properties:
    %       Name     - A name for the layer.
    %       Ceiling  - Maximum value to saturate to
    %
    %   Example:
    %       Create a clipped relu layer.
    %
    %       layer = clippedReluLayer()
    %
    %   See also clippedReluLayer
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
    
    properties(SetAccess = private, Dependent)
        Ceiling  % Maximum value to saturate to
    end
    
    methods
        function this = ClippedReLULayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
        
        function val = get.Ceiling(this)
            val = this.PrivateLayer.Ceiling;
        end
        
        function out = saveobj(this)
            out.Version = 1.0;
            out.Name = this.PrivateLayer.Name;
            out.Ceiling = this.PrivateLayer.Ceiling;
        end
    end
    
    methods(Static)
        function this = loadobj(in)
            internalLayer = nnet.internal.cnn.layer.ClippedReLU(in.Name, in.Ceiling);
            this = nnet.cnn.layer.ClippedReLULayer(internalLayer);
        end
    end
    
    methods(Access = protected)
        function [description, type] = getOneLineDisplay(obj)
            description = iGetMessageString('nnet_cnn:layer:ClippedReLULayer:oneLineDisplay', ...
                num2str(obj.Ceiling));
            
            type = iGetMessageString( 'nnet_cnn:layer:ClippedReLULayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            groups = [
                this.propertyGroupGeneral( {'Name'} )
                this.propertyGroupHyperparameters( {'Ceiling'} )
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