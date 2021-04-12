classdef CrossChannelNormalizationLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % CrossChannelNormalizationLayer   Cross channel (local response) normalization layer
    %
    %   To create a cross channel normalization layer, use
    %   crossChannelNormalizationLayer. This type of layer is also known as
    %   local response normalization.
    %
    %   A cross channel normalization layer performs channel-wise
    %   normalization. For each element in the input x, we compute a
    %   normalized value y using the following formula:
    %
    %       y = x/(K + Alpha*ss/windowChannelSize)^Beta
    %
    %   where ss is the sum of squares of the elements in the normalization
    %   window. This function can be seen as a form of lateral inhibition
    %   between channels.
    %
    %   CrossChannelNormalizationLayer properties:
    %       Name                        - A name for the layer.
    %       WindowChannelSize            - The size of the channel window for
    %                                     normalization.
    %       Alpha                       - A multiplier for normalization
    %                                     term.
    %       Beta                        - The exponent for the normalization
    %                                     term.
    %       K                           - An additive constant for the
    %                                     normalization term.
    %
    %   Example:
    %       Create a local response normalization layer for channel-wise
    %       normalization, where a window of 5 channels will be used to normalize
    %       each element, and the additive constant for the normalizer is 1.
    %
    %       layer = crossChannelNormalizationLayer(5, 'K', 1);
    %
    % [1]   A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet
    %       Classification with Deep Convolutional Neural Networks", in
    %       Advances in Neural Information Processing Systems 25, 2012.
    %
    %   See also crossChannelNormalizationLayer
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
    
    properties(SetAccess = private, Dependent)
        
        % WindowChannelSize   Size of the window for normalization
        %   The size of a window which controls the number of channels that are
        %   used for the normalization of each element. For example, if
        %   this value is 3, each element will be normalized by its
        %   neighbors in the previous channel and the next channel. If WindowChannelSize
        %   is even, then the window will be asymmetric. For example, if it
        %   is 4, each element is normalized by its neighbor in the
        %   previous channel, and by its neighbors in the next two channels.  The
        %   value must be a positive integer.
        WindowChannelSize
        
        % Alpha   Multiplier for the normalization term
        %   The Alpha term in the normalization formula.
        Alpha
        
        % Beta   Exponent for the normalization term
        %   The Beta term in the normalization formula.
        Beta
        
        % K   Additive constant for the normalization term
        %   The K term in the normalization formula.
        K
    end
    
    methods
        function this = CrossChannelNormalizationLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            out.Version = 1.0;
            out.Name = this.PrivateLayer.Name;
            out.WindowChannelSize = this.PrivateLayer.WindowChannelSize;
            out.Alpha = this.PrivateLayer.Alpha;
            out.Beta = this.PrivateLayer.Beta;
            out.K = this.PrivateLayer.K;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
        
        function val = get.WindowChannelSize(this)
            val = this.PrivateLayer.WindowChannelSize;
        end
        
        function val = get.Alpha(this)
            val = this.PrivateLayer.Alpha;
        end
        
        function val = get.Beta(this)
            val = this.PrivateLayer.Beta;
        end
        
        function val = get.K(this)
            val = this.PrivateLayer.K;
        end
    end
    
    methods(Hidden, Static)
        function this = loadobj(in)
            internalLayer = nnet.internal.cnn.layer.LocalMapNorm2D(in.Name, in.WindowChannelSize, in.Alpha, in.Beta, in.K);
            this = nnet.cnn.layer.CrossChannelNormalizationLayer(internalLayer);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            WindowChannelSizeString = int2str( this.WindowChannelSize );
            description = iGetMessageString( ...
                'nnet_cnn:layer:CrossChannelNormalizationLayer:oneLineDisplay', ...
                WindowChannelSizeString );
            
            type = iGetMessageString( 'nnet_cnn:layer:CrossChannelNormalizationLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            hyperparameters = {
                'WindowChannelSize'
                'Alpha'
                'Beta'
                'K'
                };
            
            groups = [
                this.propertyGroupGeneral( {'Name'} )
                this.propertyGroupHyperparameters( hyperparameters )
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