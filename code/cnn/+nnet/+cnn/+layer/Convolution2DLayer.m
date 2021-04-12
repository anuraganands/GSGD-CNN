classdef Convolution2DLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % Convolution2DLayer   2-D convolution layer
    %
    %   To create a convolution layer, use convolution2dLayer
    %
    %   Convolution2DLayer properties:
    %       Name                        - A name for the layer.
    %       FilterSize                  - The height and width of the
    %                                     filters.
    %       NumChannels                 - The number of channels for each
    %                                     filter.
    %       NumFilters                  - The number of filters.
    %       Stride                      - The step size for traversing the
    %                                     input vertically and
    %                                     horizontally.
    %       PaddingMode                 - The mode used to determine the
    %                                     padding.
    %       PaddingSize                 - The padding applied to the input 
    %                                     along the edges.
    %       Weights                     - Weights of the layer.
    %       Bias                        - Biases of the layer.
    %       WeightLearnRateFactor       - A number that specifies
    %                                     multiplier for the learning rate
    %                                     of the weights.
    %       BiasLearnRateFactor         - A number that specifies a
    %                                     multiplier for the learning rate
    %                                     for the biases.
    %       WeightL2Factor              - A number that specifies a
    %                                     multiplier for the L2 weight
    %                                     regulariser for the weights.
    %       BiasL2Factor                - A number that specifies a
    %                                     multiplier for the L2 weight
    %                                     regulariser for the biases.
    %
    %   Example:
    %       Create a convolution layer with 5 filters of size 10-by-10.
    %
    %       layer = convolution2dLayer(10, 5);
    %
    %   See also convolution2dLayer.
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name         
    end
    
    properties(SetAccess = private, Dependent)
        % FilterSize   The height and width of the filters
        %   The height and width of the filters. This is a row vector [h w]
        %   where h is the filter height and w is the filter width.
        FilterSize
        
        % NumChannels   The number of channels in the input
        %   The number of channels in the input. This can be set to 'auto',
        %   in which case the correct value will be determined at training
        %   time.
        NumChannels
        
        % NumFilters   The number of filters
        %   The number of filters for this layer. This also determines how
        %   many maps there will be in the output.
        NumFilters
        
        % Stride   The vertical and horizontal stride
        %   The step size for traversing the input vertically and
        %   horizontally. This is a row vector [u v] where u is the
        %   vertical stride, and v is the horizontal stride.
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
        %   The padding that is applied to the input vertically and
        %   horizontally. This a row vector [a b] where a is the padding
        %   applied to the top and bottom of the input, and b is the
        %   padding applied to the left and right of the image.
        Padding
    end
    
    properties(Dependent)
        % Weights   The weights for the layer
        %   The filters for the convolutional layer. An array with size
        %   FilterSize(1)-by-FilterSize(2)-by-NumChannels-by-NumFilters.
        Weights
        
        % Bias   The bias vector for the layer
        %   The bias for the convolutional layer. The size will be
        %   1-by-1-by-NumFilters.
        Bias
        
        % WeightLearnRateFactor   The learning rate factor for the weights
        %   The learning rate factor for the weights. This factor is
        %   multiplied with the global learning rate to determine the
        %   learning rate for the weights in this layer. For example, if it
        %   is set to 2, then the learning rate for the weights in this
        %   layer will be twice the current global learning rate.
        WeightLearnRateFactor
        
        % WeightL2Factor   The L2 regularization factor for the weights
        %   The L2 regularization factor for the weights. This factor is
        %   multiplied with the global L2 regularization setting to
        %   determine the L2 regularization setting for the weights in this
        %   layer. For example, if it is set to 2, then the L2
        %   regularization for the weights in this layer will be twice the
        %   global L2 regularization setting.
        WeightL2Factor
        
        % BiasLearnRateFactor   The learning rate factor for the biases
        %   The learning rate factor for the bias. This factor is
        %   multiplied with the global learning rate to determine the
        %   learning rate for the bias in this layer. For example, if it
        %   is set to 2, then the learning rate for the bias in this layer
        %   will be twice the current global learning rate.
        BiasLearnRateFactor
        
        % BiasL2Factor   The L2 regularization factor for the biases
        %   The L2 regularization factor for the biases. This factor is
        %   multiplied with the global L2 regularization setting to
        %   determine the L2 regularization setting for the biases in this
        %   layer. For example, if it is set to 2, then the L2
        %   regularization for the biases in this layer will be twice the
        %   global L2 regularization setting.
        BiasL2Factor
    end
    
    methods
        function this = Convolution2DLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 3.0;
            out.Name = privateLayer.Name;
            out.FilterSize = privateLayer.FilterSize;
            out.NumChannels = privateLayer.NumChannels;
            out.NumFilters = privateLayer.NumFilters;
            out.Stride = privateLayer.Stride;
            out.PaddingMode = privateLayer.PaddingMode;
            out.PaddingSize = privateLayer.PaddingSize;
            out.Weights = toStruct(privateLayer.Weights);
            out.Bias = toStruct(privateLayer.Bias);
        end

        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
        
        function val = get.Weights(this)
            val = this.PrivateLayer.Weights.HostValue;
        end
        
        function this = set.Weights(this, value)
            classes = {'single', 'double', 'gpuArray'};
            if(this.filterGroupsAreUsed())
                expectedNumChannels = iExpectedNumChannels(this.NumChannels(1));
            else
                expectedNumChannels = iExpectedNumChannels(this.NumChannels);
            end
            attributes = {'size', [this.FilterSize expectedNumChannels sum(this.NumFilters)], 'nonempty', 'real'};
            validateattributes(value, classes, attributes);
            
            % Call inferSize to determine the size of the layer
            if(this.filterGroupsAreUsed())
                inputChannels = size(value,3)*2;
            else
                inputChannels = size(value,3);
            end
            this.PrivateLayer = this.PrivateLayer.inferSize( [NaN NaN inputChannels] );
            this.PrivateLayer.Weights.Value = gather(value);
        end
        
        function val = get.Bias(this)
            val = this.PrivateLayer.Bias.HostValue;
        end
        
        function this = set.Bias(this, value)
            classes = {'single', 'double', 'gpuArray'};
            attributes = {'size', [1 1 sum(this.NumFilters)], 'nonempty', 'real'};
            validateattributes(value, classes, attributes);
            
            this.PrivateLayer.Bias.Value = gather(value);
        end
        
        function val = get.FilterSize(this)
            val = this.PrivateLayer.FilterSize;
        end
        
        function val = get.NumChannels(this)
            val = this.PrivateLayer.NumChannels;
            if(this.filterGroupsAreUsed())
                val = [val val];
            end
            if isempty(val)
                val = 'auto';
            end
        end
        
        function val = get.NumFilters(this)
            val = this.PrivateLayer.NumFilters;
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
        
        function val = get.WeightLearnRateFactor(this)
            val = this.PrivateLayer.Weights.LearnRateFactor;
        end
        
        function this = set.WeightLearnRateFactor(this, value)
            iAssertValidFactor(value,'WeightLearnRateFactor');
            this.PrivateLayer.Weights.LearnRateFactor = value;
        end
        
        function val = get.BiasLearnRateFactor(this)
            val = this.PrivateLayer.Bias.LearnRateFactor;
        end
        
        function this = set.BiasLearnRateFactor(this, value)
            iAssertValidFactor(value,'BiasLearnRateFactor');
            this.PrivateLayer.Bias.LearnRateFactor = value;
        end
        
        function val = get.WeightL2Factor(this)
            val = this.PrivateLayer.Weights.L2Factor;
        end
        
        function this = set.WeightL2Factor(this, value)
            iAssertValidFactor(value,'WeightL2Factor');
            this.PrivateLayer.Weights.L2Factor = value;
        end
        
        function val = get.BiasL2Factor(this)
            val = this.PrivateLayer.Bias.L2Factor;
        end
        
        function this = set.BiasL2Factor(this, value)
            iAssertValidFactor(value,'BiasL2Factor');
            this.PrivateLayer.Bias.L2Factor = value;
        end
    end
    
    methods
        function val = get.Padding(this)
            % This is required for backward compatibility.
            iValidatePaddingCanBeExpressedAs1By2Vector( ...
                this.PrivateLayer.PaddingSize);
            val = [this.PrivateLayer.PaddingSize(1) ...
                this.PrivateLayer.PaddingSize(3)];
            warning(message('nnet_cnn:layer:Convolution2DLayer:PaddingObsolete'));
        end
    end
    
    methods(Static)
        function this = loadobj(in)
            if in.Version <= 1
                in = iUpgradeVersionOneToVersionTwo(in);
            end
            if in.Version <= 2
                in = iUpgradeVersionTwoToVersionThree(in);
            end
            this = iLoadConvolution2DLayerFromCurrentVersion(in);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            numFiltersString = int2str( sum(this.NumFilters) );
            filterSizeString = i2DSizeToString( this.FilterSize );
            if ~isequal(this.NumChannels, 'auto')
                % When using filter groups, the number of channels is
                % replicated to match NumFilters. For display, only show
                % the first element.
                numChannelsString = ['x' int2str( this.NumChannels(1) )];
            else
                numChannelsString = '';
            end
            strideString = "["+int2str( this.Stride )+"]";
            
            if this.PaddingMode ~= "manual"
                paddingSizeString = "'"+this.PaddingMode+"'";
            else
                paddingSizeString = "["+int2str( this.PaddingSize )+"]";
            end
            
            description = iGetMessageString( ...
                'nnet_cnn:layer:Convolution2DLayer:oneLineDisplay', ...
                numFiltersString, ...
                filterSizeString, ...
                numChannelsString, ...
                strideString, ...
                paddingSizeString );
            
            type = iGetMessageString( 'nnet_cnn:layer:Convolution2DLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            hyperparameters = {
                'FilterSize'
                'NumChannels'
                'NumFilters'
                'Stride'
                'PaddingMode'
                'PaddingSize'
                };
            
            learnableParameters = {'Weights', 'Bias'};
            
            groups = [
                this.propertyGroupGeneral( {'Name'} )
                this.propertyGroupHyperparameters( hyperparameters )
                this.propertyGroupLearnableParameters( learnableParameters )
                ];
        end
        
        function footer = getFooter( this )
            variableName = inputname(1);
            footer = this.createShowAllPropertiesFooter( variableName );
        end
        
        function tf = filterGroupsAreUsed(this)
            tf = numel(this.NumFilters) ~= 1;
        end
    end
end

function iValidatePaddingCanBeExpressedAs1By2Vector(paddingSize)
if(iPaddingCanBeExpressedAs1By2Vector(paddingSize))
else
    error(message('nnet_cnn:layer:Convolution2DLayer:PaddingCannotBeAsymmetric'));
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

function iAssertValidFactor(value,factorName)
try
    nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value,factorName);
catch e
    % Reduce the stack trace of the error message by throwing as caller
    throwAsCaller(e)
end
end

function S = iUpgradeVersionOneToVersionTwo(S)
% iUpgradeVersionOneToVersionTwo   Upgrade a v1 (2016a) saved struct to a v2 saved struct
%   This means gathering the bias and weights from the GPU and putting them
%   on the host.

S.Version = 2;
try
    S.Weights.Value = gather(S.Weights.Value);
    S.Bias.Value = gather(S.Bias.Value);
catch e
    % Only throw the error we want to throw.
    e = MException( message('nnet_cnn:layer:Convolution2DLayer:MustHaveGPUToLoadFrom2016a'));
    throwAsCaller(e);
end
end

function S = iUpgradeVersionTwoToVersionThree(S)
S.Version = 3;
S.PaddingMode = 'manual';
S.PaddingSize = iCalculatePaddingSize(S.Padding);
end

function obj = iLoadConvolution2DLayerFromCurrentVersion(in)
internalLayer = nnet.internal.cnn.layer.Convolution2D( ...
    in.Name, in.FilterSize, in.NumChannels, ...
    in.NumFilters, in.Stride, in.PaddingMode, in.PaddingSize);
internalLayer.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Weights);
internalLayer.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Bias);

obj = nnet.cnn.layer.Convolution2DLayer(internalLayer);
end

function expectedNumChannels = iExpectedNumChannels(NumChannels)
expectedNumChannels = NumChannels;
if isequal(expectedNumChannels, 'auto')
    expectedNumChannels = NaN;
end
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end

function paddingSize = iCalculatePaddingSize(padding)
paddingSize = nnet.internal.cnn.layer.padding.calculatePaddingSize(padding);
end