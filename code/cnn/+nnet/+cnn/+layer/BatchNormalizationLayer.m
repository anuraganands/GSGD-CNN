classdef BatchNormalizationLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % BatchNormalizationLayer  Batch Normalization layer
    %
    % To create a batch normalization layer, use batchNormalizationLayer.
    %
    % A layer which normalizes each channel across a mini-batch. This can
    % be useful in reducing sensitivity to model hyperparameters and
    % improving training time.
    %
    % BatchNormalization properties:
    %     Name - A name for the layer
    %     NumChannels - The number of channels in the layer input
    %     Epsilon - Offset for the divisor to prevent divide-by-zero errors.
    %               Must be at least 1e-5.
    %
    % Properties for learnable parameters:
    %     Offset - Offset per channel (beta in original paper).
    %     OffsetLearnRateFactor - Multiplier for the learning rate for Offset
    %     OffsetL2Factor - Multiplier for the L2 weight regulariser for Offset.
    %
    %     Scale - Scale per channel (gamma in original paper).
    %     ScaleLearnRateFactor - Multiplier for the learning rate for Scale.
    %     ScaleL2Factor - Multiplier for the L2 weight regulariser for Scale.
    %
    % Properties determined during training:
    %     TrainedMean - Per-channel mean of the layer input data.
    %     TrainedVariance - Per-channel mean of the layer input data.
    %
    % Example:
    %     Create a batch normalization layer.
    %
    %     layer = batchNormalizationLayer()
    %
    % See also batchNormalizationLayer
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
    
    properties(SetAccess = private, Dependent)
        % NumChannels   The number of channels in the input. This can be
        % set to 'auto', in which case the correct value will be determined
        % at training time.
        NumChannels
        
        % TrainedMean  Per-channel mean of the layer input data, determined
        % during training.
        TrainedMean
        % TrainedVariance  Per-channel variance of the layer input data,
        % determined during training.
        TrainedVariance
    end
    
    properties(Dependent)
        %Epsilon  Offset for the divisor to prevent divide-by-zero errors
        Epsilon
        
        %Offset  Bias per channel
        Offset
        %OffsetLearnRateFactor  Multiplier for the learning rate for Offset
        OffsetLearnRateFactor
        %OffsetL2Factor  Multiplier for the L2 weight regulariser for Offset
        OffsetL2Factor
        
        %Scale  Scale per channel
        Scale
        %ScaleLearnRateFactor  Multiplier for the learning rate for Scale
        ScaleLearnRateFactor
        %ScaleL2Factor  Multiplier for the L2 weight regulariser for Scale
        ScaleL2Factor
    end
    
    methods
        function this = BatchNormalizationLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
        
        function val = get.NumChannels(this)
            val = this.PrivateLayer.NumChannels;
            if isempty(val)
                val = 'auto';
            end
        end
        
        function val = get.TrainedMean(this)
            val = this.PrivateLayer.TrainedMean;
        end
        
        function val = get.TrainedVariance(this)
            val = this.PrivateLayer.TrainedVariance;
        end
        
        function val = get.Epsilon(this)
            val = this.PrivateLayer.Epsilon;
        end
        function this = set.Epsilon(this, val)
            minEpsilon = 1e-5; % Defined by cuDNN
            validateattributes(val, {'numeric'}, ...
                {'scalar','finite','real','nonnegative','>=',minEpsilon});
            this.PrivateLayer.Epsilon = val;
        end
        
        function val = get.Offset(this)
            val = this.PrivateLayer.Offset.Value;
        end
        function this = set.Offset(this, val)
            classes = {'single', 'double', 'gpuArray'};
            expectedNumChannels = iExpectedNumChannels(this.NumChannels);
            attributes = {'size', [1 1 expectedNumChannels], 'nonempty', 'real'};
            validateattributes(val, classes, attributes);
            
            % Call inferSize to determine the size of the layer
            inputChannels = size(val,3);
            this.PrivateLayer = this.PrivateLayer.inferSize( [NaN NaN inputChannels] );
            this.PrivateLayer.Offset.Value = gather(val);
        end
        
        function val = get.OffsetLearnRateFactor(this)
            val = this.PrivateLayer.Offset.LearnRateFactor;
        end
        function this = set.OffsetLearnRateFactor(this, val)
            iAssertValidFactor(val,'OffsetLearnRateFactor');
            this.PrivateLayer.Offset.LearnRateFactor = val;
        end
        
        function val = get.OffsetL2Factor(this)
            val = this.PrivateLayer.Offset.L2Factor;
        end
        function this = set.OffsetL2Factor(this, val)
            iAssertValidFactor(val,'OffsetL2RateFactor');
            this.PrivateLayer.Offset.L2Factor = val;
        end
        
        function val = get.Scale(this)
            val = this.PrivateLayer.Scale.Value;
        end
        function this = set.Scale(this, val)
            classes = {'single', 'double', 'gpuArray'};
            expectedNumChannels = iExpectedNumChannels(this.NumChannels);
            attributes = {'size', [1 1 expectedNumChannels], 'nonempty', 'real'};
            validateattributes(val, classes, attributes);
            
            % Call inferSize to determine the size of the layer
            inputChannels = size(val,3);
            this.PrivateLayer = this.PrivateLayer.inferSize( [NaN NaN inputChannels] );
            this.PrivateLayer.Scale.Value = gather(val);
        end
        
        function val = get.ScaleLearnRateFactor(this)
            val = this.PrivateLayer.Scale.LearnRateFactor;
        end
        function this = set.ScaleLearnRateFactor(this, val)
            iAssertValidFactor(val,'ScaleLearnRateFactor');
            this.PrivateLayer.Scale.LearnRateFactor = val;
        end
        
        function val = get.ScaleL2Factor(this)
            val = this.PrivateLayer.Scale.L2Factor;
        end
        function this = set.ScaleL2Factor(this, val)
            iAssertValidFactor(val,'ScaleL2RateFactor');
            this.PrivateLayer.Scale.L2Factor = val;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 1.0;
            out.Name = privateLayer.Name;
            out.Epsilon = privateLayer.Epsilon;
            out.NumChannels = privateLayer.NumChannels;
            out.TrainedMean = privateLayer.TrainedMean;
            out.TrainedVariance = privateLayer.TrainedVariance;
            out.NumTrainingSamples = privateLayer.NumTrainingSamples;
            out.Offset = toStruct(privateLayer.Offset);
            out.Scale = toStruct(privateLayer.Scale);
        end
        
    end
    
    methods(Static)
        function this = loadobj(in)
            internalLayer = nnet.internal.cnn.layer.BatchNormalization(in.Name, in.NumChannels, in.Epsilon);
            internalLayer.Offset = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Offset);
            internalLayer.Scale = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Scale);
            internalLayer.TrainedMean = in.TrainedMean;
            internalLayer.TrainedVariance = in.TrainedVariance;
            internalLayer.NumTrainingSamples = in.NumTrainingSamples;
            
            this = nnet.cnn.layer.BatchNormalizationLayer(internalLayer);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(obj)
            if (obj.PrivateLayer.HasSizeDetermined)
                description = iGetMessageString( ...
                    'nnet_cnn:layer:BatchNormalizationLayer:oneLineDisplay', ...
                    num2str(obj.NumChannels));
            else
                description = iGetMessageString( ...
                    'nnet_cnn:layer:BatchNormalizationLayer:oneLineDisplayNoChannels' );
            end
            
            type = iGetMessageString( 'nnet_cnn:layer:BatchNormalizationLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            groups = [
                this.propertyGroupGeneral( {'Name', 'NumChannels', 'TrainedMean', 'TrainedVariance'} )
                this.propertyGroupHyperparameters( {'Epsilon'} )
                this.propertyGroupLearnableParameters( {'Offset', 'Scale'} )
                ];
        end
        
        function footer = getFooter( this )
            variableName = inputname(1);
            footer = this.createShowAllPropertiesFooter( variableName );
        end
        
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function expectedNumChannels = iExpectedNumChannels(NumChannels)
expectedNumChannels = NumChannels;
if isequal(expectedNumChannels, 'auto')
    expectedNumChannels = NaN;
end
end

function iAssertValidFactor(value,factorName)
try
    nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value,factorName);
catch e
    % Reduce the stack trace of the error message by throwing as caller
    throwAsCaller(e)
end
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end