classdef RegressionOutputLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % RegressionOutputLayer   Regression output layer
    %
    %   To create a regression output layer, use regressionLayer.
    %
    %   A regression output layer. This layer is used as the output for
    %   a network that performs regression.
    %
    %   RegressionOutputLayer properties:
    %       Name                        - A name for the layer.
    %       ResponseNames               - The names of the responses.
    %       LossFunction                - The loss function that is used
    %                                     for training.
    %
    %   Example:
    %       Create a regression output layer.
    %
    %       layer = regressionLayer();
    %
    %   See also regressionLayer
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name         
    end
    
    properties(SetAccess = private, Dependent)
        % ResponseNames   The names of the responses
        %   A cell array containing the names of the responses. This will
        %   be automatically determined at training time. Prior to
        %   training, it will be empty.
        ResponseNames
    end
    
    properties(SetAccess = private)
        % LossFunction   The loss function for training
        %   The loss function that will be used during training. Possible
        %   values are:
        %       'mean-squared-error'    - Mean squared error
        LossFunction = 'mean-squared-error';
    end
    
    methods
        function this = RegressionOutputLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 2.0;
            out.Name = privateLayer.Name;
            out.ResponseNames = privateLayer.ResponseNames;
            out.ObservationDim = privateLayer.ObservationDim;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
        
        function val = get.ResponseNames(this)
            val = this.PrivateLayer.ResponseNames;
        end
    end
    
    methods(Hidden, Static)
        function this = loadobj(in)
            if in.Version <= 1
                in = iUpgradeFromVersionOneToVersionTwo(in);
            end
            internalLayer = nnet.internal.cnn.layer.MeanSquaredError.constructWithObservationDim( ...
                in.Name, in.ResponseNames, in.ObservationDim );
            this = nnet.cnn.layer.RegressionOutputLayer(internalLayer);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            lossFunction = this.LossFunction;
            
            numResponses = numel(this.ResponseNames);
            
            if numResponses==0
                description = iGetMessageString( ...
                    'nnet_cnn:layer:RegressionOutputLayer:oneLineDisplayNoResponses', ....
                    lossFunction );
            elseif numResponses==1
                description = iGetMessageString( ...
                    'nnet_cnn:layer:RegressionOutputLayer:oneLineDisplayOneResponse', ....
                    lossFunction, ...
                    this.ResponseNames{1} );
            elseif numResponses==2
                description = iGetMessageString( ...
                    'nnet_cnn:layer:RegressionOutputLayer:oneLineDisplayTwoResponses', ....
                    lossFunction, ...
                    this.ResponseNames{1}, ...
                    this.ResponseNames{2} );
            elseif numResponses>=3
                description = iGetMessageString( ...
                    'nnet_cnn:layer:RegressionOutputLayer:oneLineDisplayNResponses', ....
                    lossFunction, ...
                    this.ResponseNames{1}, ...
                    this.ResponseNames{2}, ...
                    int2str( numResponses-2 ) );
            end
            
            type = iGetMessageString( 'nnet_cnn:layer:RegressionOutputLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            generalParameters = {
                'Name'
                'ResponseNames'
                };
            
            groups = [
                this.propertyGroupGeneral( generalParameters )
                this.propertyGroupHyperparameters( {'LossFunction'} )
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

function S = iUpgradeFromVersionOneToVersionTwo( S )
% iUpgradeFromVersionOneToVersionTwo   Add the observation dimension
% property to v1 layers. All v1 layers, created from R2017b and before,
% have observation dimension equal to 4, corresponding to the image data
% format.
S.Version = 2.0;
S.ObservationDim = 4;
end