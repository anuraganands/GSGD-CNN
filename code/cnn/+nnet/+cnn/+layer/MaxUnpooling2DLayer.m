classdef MaxUnpooling2DLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % MaxUnpooling2DLayer   Max unpooling layer
    %
    %   To create a 2d max unpooling layer, use maxUnpooling2dLayer.
    %
    %   A max unpooling layer. This layer unpools the output of a max
    %   pooling layer.
    %
    %   MaxUnpooling2DLayer properties:
    %       Name                    - A name for the layer.
    %
    %   Example:
    %       Create a max unpooling layer to unpool output of a max pooling
    %       layer:
    %
    %       layer = maxUnpooling2dLayer();
    %
    %   See also maxUnpooling2dLayer, maxPooling2dLayer.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end       
    
    methods
        function this = MaxUnpooling2DLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 1.0;
            out.Name = privateLayer.Name;            
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
            internalLayer = nnet.internal.cnn.layer.MaxUnpooling2D(in.Name);
            this = nnet.cnn.layer.MaxUnpooling2DLayer(internalLayer);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(~)          
            description = iGetMessageString(...
                'nnet_cnn:layer:MaxUnpooling2DLayer:Type');
            
            type = iGetMessageString(...
                'nnet_cnn:layer:MaxUnpooling2DLayer:Type');
        end
        
        function groups = getPropertyGroups( this )                       
            groups = this.propertyGroupGeneral( {'Name'} );                
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end