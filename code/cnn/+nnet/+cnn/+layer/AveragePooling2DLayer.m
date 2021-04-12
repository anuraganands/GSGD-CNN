classdef AveragePooling2DLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % AveragePooling2DLayer   Average pooling layer
    %
    %   To create a 2d average pooling layer, use averagePooling2dLayer
    %
    %   An average pooling layer. This layer performs downsampling.
    %
    %   AveragePooling2DLayer properties:
    %       Name                    - A name for the layer.
    %       PoolSize                - The height and width of pooling
    %                                 regions.
    %       Stride                  - The vertical and horizontal stride.
    %       PaddingMode             - The mode used to determine the
    %                                 padding.
    %       PaddingSize             - The padding applied to the input 
    %                                 along the edges.
    %
    %   Example:
    %       Create an average pooling layer with non-overlapping pooling
    %       regions, which down-samples by a factor of 2:
    %
    %       layer = averagePooling2dLayer(2, 'Stride', 2);
    %
    %   See also averagePooling2dLayer
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
    
    properties(SetAccess = private, Dependent)
        % PoolSize   The height and width of a pooling region
        %   The size the pooling regions. This is a vector [h w] where h is
        %   the height of a pooling region, and w is the width of a pooling
        %   region.
        PoolSize
        
        % Stride   The vertical and horizontal stride
        %   The step size for traversing the input vertically and
        %   horizontally. This is a vector [u v] where u is the vertical
        %   stride and v is the horizontal stride.
        Stride
        
        % PaddingMode   The mode used to determine the padding
        %   The mode used to calculate the PaddingSize property. This can
        %   be:
        %       'manual'    - PaddingSize is specified manually. 
        %       'same'      - PaddingSize is calculated so that the output
        %                     is the same size as the input when the stride
        %                     is 1. More generally, the output size will be
        %                     ceil(inputSize/stride), where inputSize is 
        %                     the height and width of the input.
        PaddingMode
        
        % PaddingSize   The padding applied to the input along the edges
        %   The padding that is applied along the edges. This is a row 
        %   vector [t b l r] where t is the padding to the top, b is the 
        %   padding applied to the bottom, l is the padding applied to the 
        %   left, and r is the padding applied to the right.
        PaddingSize
    end
    
    properties(SetAccess = private, Dependent, Hidden)
        % Padding   The vertical and horizontal padding
        %   Padding property will be removed in a future release. Use 
        %   PaddingSize instead.
        %
        %   The padding that is applied to the input for this layer. This
        %   is a vector [a b] where a is the padding applied to the top and
        %   bottom of the input, and b is the padding applied to the left
        %   and right.
        Padding
    end
    
    methods
        function this = AveragePooling2DLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;            
            out.Version = 2.0;
            out.Name = privateLayer.Name;
            out.PoolSize = privateLayer.PoolSize;
            out.Stride = privateLayer.Stride;
            out.PaddingMode = privateLayer.PaddingMode;
            out.PaddingSize = privateLayer.PaddingSize;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end        
        
        function val = get.PoolSize(this)
            val = this.PrivateLayer.PoolSize;
        end
        
        function val = get.Stride(this)
            val = this.PrivateLayer.Stride;
        end
        
        function val = get.PaddingMode(this)
            val = this.PrivateLayer.PaddingMode;
        end
        
        function val = get.PaddingSize(this)
            val = this.PrivateLayer.PaddingSize;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
    end
    
    methods
        function val = get.Padding(this)
            % This is required for backward compatibility.
            iValidatePaddingCanBeExpressedAs1By2Vector( ...
                this.PrivateLayer.PaddingSize);
            val = [this.PrivateLayer.PaddingSize(1) ...
                this.PrivateLayer.PaddingSize(3)];
            warning(message('nnet_cnn:layer:AveragePooling2DLayer:PaddingObsolete'));
        end
    end
    
    methods(Hidden, Static)
        function this = loadobj(in)
            if in.Version <= 1
                in = iUpgradeVersionOneToVersionTwo(in);
            end
            this = iLoadAveragePooling2DLayerFromCurrentVersion(in);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            poolSizeString = i2DSizeToString( this.PoolSize );
            strideString = "["+int2str( this.Stride )+"]";
            
            if this.PaddingMode ~= "manual"
                paddingSizeString = "'"+this.PaddingMode+"'";
            else
                paddingSizeString = "["+int2str( this.PaddingSize )+"]";
            end
            
            description = iGetMessageString(  ...
                'nnet_cnn:layer:AveragePooling2DLayer:oneLineDisplay', ...
                poolSizeString, ...
                strideString, ...
                paddingSizeString );
            
            type = iGetMessageString( 'nnet_cnn:layer:AveragePooling2DLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            hyperparameters = {
                'PoolSize'
                'Stride'
                'PaddingMode'
                'PaddingSize'
                };
            
            groups = [
                this.propertyGroupGeneral( {'Name'} )
                this.propertyGroupHyperparameters( hyperparameters )
                ];
        end
    end
end

function iValidatePaddingCanBeExpressedAs1By2Vector(paddingSize)
if(iPaddingCanBeExpressedAs1By2Vector(paddingSize))
else
    error(message('nnet_cnn:layer:AveragePooling2DLayer:PaddingCannotBeAsymmetric'));
end
end

function tf = iPaddingCanBeExpressedAs1By2Vector(paddingSize)
tf = (paddingSize(1) == paddingSize(2)) && (paddingSize(3) == paddingSize(4));
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function sizeString = i2DSizeToString( sizeVector )
% i2DSizeToString   Convert a 2-D size stored in a vector of 2 elements
% into a string separated by 'x'.
sizeString = [ ...
    int2str( sizeVector(1) ) ...
    'x' ...
    int2str( sizeVector(2) ) ];
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end

function S = iUpgradeVersionOneToVersionTwo(S)
S.Version = 2;
S.PaddingMode = 'manual';
S.PaddingSize = iCalculatePaddingSize(S.Padding);
end

function obj = iLoadAveragePooling2DLayerFromCurrentVersion(in)
internalLayer = nnet.internal.cnn.layer.AveragePooling2D( ...
    in.Name, ...
    in.PoolSize, ...
    in.Stride, ...
    in.PaddingMode, ...
    in.PaddingSize);
obj = nnet.cnn.layer.AveragePooling2DLayer(internalLayer);
end

function paddingSize = iCalculatePaddingSize(padding)
paddingSize = nnet.internal.cnn.layer.padding.calculatePaddingSize(padding);
end