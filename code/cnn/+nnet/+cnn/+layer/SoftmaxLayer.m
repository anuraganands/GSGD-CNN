classdef SoftmaxLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % SoftmaxLayer   Softmax layer
    %
    %   A softmax layer. This layer is useful for classification problems.
    %
    %   SoftmaxLayer properties:
    %       Name                    - A name for the layer
    %
    %   Example:
    %       Create a softmax layer.
    %
    %       layer = softmaxLayer();
    %
    %   See also softmaxLayer
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
    
    methods
        function this = SoftmaxLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            out.Version = 2.0;
            out.Name = this.PrivateLayer.Name;
            out.VectorFormat = this.PrivateLayer.VectorFormat;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
    end
    
    methods(Hidden, Static)
        function this = loadobj(in)
            if in.Version <= 1
                in = iUpgradeVersionOneToVersionTwo(in);
            end
            this = iLoadSoftmaxLayerFromCurrentVersion(in);
        end
    end
    
    methods(Access = protected)
        function [description, type] = getOneLineDisplay(~)
            description = iGetMessageString('nnet_cnn:layer:SoftmaxLayer:oneLineDisplay');
            
            type = iGetMessageString( 'nnet_cnn:layer:SoftmaxLayer:Type' );
        end
    end
end

function messageString = iGetMessageString( varargin )
    messageString = getString( message( varargin{:} ) );
end

function S = iUpgradeVersionOneToVersionTwo(S)
    % iUpgradeVersionOneToVersionTwo   Upgrade a v1 (2016a-2017a) saved struct
    % to a v2 saved struct. This means adding a "VectorFormat" property.
    
    S.Version = 2;
    S.VectorFormat = false;
end

function layer = iLoadSoftmaxLayerFromCurrentVersion(in)
internalLayer = nnet.internal.cnn.layer.Softmax(in.Name);
internalLayer.VectorFormat = in.VectorFormat;
internalLayer = internalLayer.setupForHostPrediction();
layer = nnet.cnn.layer.SoftmaxLayer(internalLayer);
end

function iAssertValidLayerName(name)
    nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end