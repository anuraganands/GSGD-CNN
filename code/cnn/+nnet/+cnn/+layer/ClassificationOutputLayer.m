classdef ClassificationOutputLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % ClassificationOutputLayer   Classification output layer
    %
    %   To create a classification output layer, use classificationLayer
    %
    %   A classification output layer. This layer is used as the output for
    %   a network that performs classification.
    %
    %   ClassificationOutputLayer properties:
    %       Name                        - A name for the layer.
    %       ClassNames                  - The names of the classes.
    %       OutputSize                  - The size of the output.
    %       LossFunction                - The loss function that is used
    %                                     for training.
    %
    %   Example:
    %       Create a classification output layer.
    %
    %       layer = classificationLayer();
    %
    %   See also classificationLayer
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
    
    properties(SetAccess = private, Dependent)
        % ClassNames   The names of the classes
        %   A cell array containing the names of the classes. This will be
        %   automatically determined at training time. Prior to training,
        %   it will be empty.
        ClassNames
        
        % OutputSize   The size of the output
        %   The size of the output. This will be determined at training
        %   time. Prior to training, it is set to 'auto'.
        OutputSize
    end
    
    properties(SetAccess = private)
        % LossFunction   The loss function for training
        %   The loss function that will be used during training. Possible
        %   values are:
        %       'crossentropyex'    - Cross-entropy for exclusive outputs.
        LossFunction = 'crossentropyex';
    end
    
    methods
        function this = ClassificationOutputLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 3.0;
            out.Name = privateLayer.Name;
            out.ObservationDim = privateLayer.ObservationDim;
            out.NumClasses = privateLayer.NumClasses;            
            out.Categories = privateLayer.Categories;
        end
        
        function val = get.OutputSize(this)
            if(isempty(this.PrivateLayer.NumClasses))
                val = 'auto';
            else
                val = this.PrivateLayer.NumClasses;
            end
        end
        
        function val = get.ClassNames(this)
            val = this.PrivateLayer.ClassNames(:);
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
                in = iUpgradeFromVersionOneToVersionTwo(in);
            end
            if in.Version <= 2
                in = iUpgradeFromVersionTwoToVersionThree(in);
            end            
            internalLayer = nnet.internal.cnn.layer.CrossEntropy.constructWithObservationDim( ...
                in.Name, in.NumClasses, in.Categories, in.ObservationDim );
            this = nnet.cnn.layer.ClassificationOutputLayer(internalLayer);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            lossFunction = this.LossFunction;
            
            numClasses = numel(this.ClassNames);
            
            if numClasses==0
                description = iGetMessageString( ...
                    'nnet_cnn:layer:ClassificationOutputLayer:oneLineDisplayNoClasses', ....
                    lossFunction );
            elseif numClasses==1
                description = iGetMessageString( ...
                    'nnet_cnn:layer:ClassificationOutputLayer:oneLineDisplayOneClass', ....
                    lossFunction, ...
                    this.ClassNames{1} );
            elseif numClasses==2
                description = iGetMessageString( ...
                    'nnet_cnn:layer:ClassificationOutputLayer:oneLineDisplayTwoClasses', ....
                    lossFunction, ...
                    this.ClassNames{1}, ...
                    this.ClassNames{2} );
            elseif numClasses>=3
                description = iGetMessageString( ...
                    'nnet_cnn:layer:ClassificationOutputLayer:oneLineDisplayNClasses', ....
                    lossFunction, ...
                    this.ClassNames{1}, ...
                    int2str( numClasses-1 ) );
            end
            
            type = iGetMessageString( 'nnet_cnn:layer:ClassificationOutputLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            if numel(this.ClassNames) < 11 && ~ischar(this.ClassNames)
                propertyList = struct;
                propertyList.Name = this.Name;
                propertyList.ClassNames = this.ClassNames';
                propertyList.OutputSize = this.OutputSize;
                groups = [
                    matlab.mixin.util.PropertyGroup(propertyList, '');
                    this.propertyGroupHyperparameters( {'LossFunction'} )
                    ];
            else
                generalParameters = {'Name' 'ClassNames' 'OutputSize'};
                groups = [
                    this.propertyGroupGeneral( generalParameters )
                    this.propertyGroupHyperparameters( {'LossFunction'} )
                    ];
            end
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
% property to v1 layers. All v1 layers, created from R2017a and before,
% have observation dimension equal to 4, corresponding to the image data
% format. A NumClasses property is also created for construction of the
% internal layer.
S.Version = 2.0;
S.ObservationDim = 4;
if isempty( S.OutputSize )
    S.NumClasses = S.OutputSize;
else
    S.NumClasses = S.OutputSize(end);
end
end

function S = iUpgradeFromVersionTwoToVersionThree( S )
% iUpgradeFromVersionTwoToVersionThree   In all layers with version <= 2
% the property ClassNames has to be replaced with Categories in 
% version 2. Here set Categories to categorical array with categories as 
% ClassNames and ordinality false.
S.Version = 3.0;
% Make sure there are no duplicate names
classNames = iRenameDuplicated(S.ClassNames);
S.Categories = categorical(classNames, classNames);
end

function renamed = iRenameDuplicated(names)
    % Makes a list of unique names, avoiding using the names that were
    % duplicated

    % Generate list of duplicated names
    [~,idx] = unique(names);    
    idx = setdiff(1:numel(names), idx);
    if ~isempty(idx)
        % Print warning 
        warning(message('nnet_cnn:layer:ClassificationOutputLayer:RenamingClassNames'));    
        duplicated = unique(names(idx));    
        renamed = matlab.lang.makeUniqueStrings( ...
            names, duplicated(:));
    else
        renamed = names;
    end
end