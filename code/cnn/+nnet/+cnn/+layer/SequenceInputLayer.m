classdef SequenceInputLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % SequenceInputLayer   Sequence input layer
    %
    %   To create a sequence input layer, use sequenceInputLayer
    %    
    %   SequenceInputLayer properties:
    %       Name                        - A name for the layer.
    %       InputSize                   - The size of the input
    %
    %   Example:
    %       Create a sequence input layer to accept a multi-dimensional
    %       time series with 5 values per timestep
    %
    %       layer = sequenceInputLayer(5)
    %
    %   See also sequenceInputLayer    
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
        
    properties(SetAccess = private, Dependent)
        % InputSize   Size of the input data as an integer.
        InputSize        
    end    
    
    methods
        function this = SequenceInputLayer(privateLayer)
            this.PrivateLayer = privateLayer;            
        end
        
        function out = saveobj(this)
            out.Version = 1.0;
            out.Name = this.PrivateLayer.Name;
            out.InputSize = this.PrivateLayer.InputSize;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
        
        function val = get.InputSize(this)
            val = this.PrivateLayer.InputSize;
        end       
    end
    
    methods(Static)                
        function this = loadobj(in)
            internalLayer = nnet.internal.cnn.layer.SequenceInput( ...
                in.Name, in.InputSize );
            this = nnet.cnn.layer.SequenceInputLayer(internalLayer);
        end        
    end    
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            inputSizeString = int2str( this.InputSize );
            description = iGetMessageString( ...
                    'nnet_cnn:layer:SequenceInputLayer:oneLineDisplay', ....
                    inputSizeString );
            
            type = iGetMessageString( 'nnet_cnn:layer:SequenceInputLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            generalParameters = {
                'Name'
                'InputSize'
                };
           
            groups = this.propertyGroupGeneral( generalParameters );
        end          
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end