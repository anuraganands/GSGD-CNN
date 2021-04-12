classdef LeakyReLULayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % LeakyReLULayer   Leaky Rectified linear unit (ReLU) layer
    %
    %   To create a leaky rectified linear unit layer, use leakyReluLayer
    %
    %   A leaky rectified linear unit layer. This type of layer performs a
    %   simple thresholding operation, where any input value less than zero
    %   is multiplied by a scalar multiple. This is equivalent to:
    %     out = in;        % For in>0
    %     out = scale.*in; % For in<=0
    %
    %   LeakyReLULayer properties:
    %       Name   - A name for the layer.
    %       Scale  - Scalar multiplier for negative values (default 0.01)
    %
    %   Example:
    %       Create a leaky relu layer.
    %
    %       layer = leakyReluLayer()
    %
    %   See also leakyReluLayer
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name         
    end
    
    properties(SetAccess = private, Dependent)
        Scale  % Scalar multiplier for negative values (default 0.01)
    end
    
    methods
        function this = LeakyReLULayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end        
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
        
        function val = get.Scale(this)
            val = this.PrivateLayer.Scale;
        end
        
        function out = saveobj(this)
            out.Version = 1.0;
            out.Name = this.PrivateLayer.Name;
            out.Scale = this.PrivateLayer.Scale;
        end
    end
    
    methods(Static)        
        function this = loadobj(in)
            internalLayer = nnet.internal.cnn.layer.LeakyReLU(in.Name, in.Scale);
            this = nnet.cnn.layer.LeakyReLULayer(internalLayer);
        end
    end

    methods(Access = protected)
        function [description, type] = getOneLineDisplay(obj)
            description = iGetMessageString('nnet_cnn:layer:LeakyReLULayer:oneLineDisplay', ...
                num2str(obj.Scale));
            
            type = iGetMessageString( 'nnet_cnn:layer:LeakyReLULayer:Type' );
        end
                        
        function groups = getPropertyGroups( this )
            groups = [
                this.propertyGroupGeneral( {'Name'} )
                this.propertyGroupHyperparameters( {'Scale'} )
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