classdef ImageInputLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % ImageInputLayer   Image input layer
    %
    %   To create an image input layer, use imageInputLayer
    %
    %   ImageInputLayer properties:
    %       Name                        - A name for the layer.
    %       InputSize                   - The size of the input
    %       DataAugmentation            - The use of the DataAugentation 
    %                                     property is not recommended. Use
    %                                     augmentedImageSource instead.
    %       Normalization               - normalization applied to image
    %                                     data every time data is forward
    %                                     propagated through the input layer
    %
    %   Example:
    %       Create an image input layer to accept color images of size 28
    %       by 28.
    %
    %       layer = imageInputLayer([28 28 3])
    %
    %   See also imageInputLayer
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
    
    properties(SetAccess = private, Dependent)
        % InputSize Size of the input image as [height, width, channels].
        InputSize
        
        % DataAugmentation    DataAugmentation is not recommended. Use
        %                     augmentedImageSource instead.
        DataAugmentation
        
        % Normalization  A string that specifies the normalization applied
        %                to image data every time data is forward
        %                propagated through the input layer. Valid values
        %                are 'zerocenter' or 'none'. This property is
        %                read-only.
        Normalization
    end
    
    methods
        function this = ImageInputLayer(privateLayer)
        this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
        out.Version = 1.0;
        out.Name = this.PrivateLayer.Name;
        out.InputSize = this.PrivateLayer.InputSize;
        out.Normalization = iSaveTransforms(this.PrivateLayer.Transforms);
        out.Augmentations = iSaveTransforms(this.PrivateLayer.TrainTransforms);
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
        
        function val = get.Normalization(this)
        if isempty(this.PrivateLayer.Transforms)
            val = 'none';
        else
            val = this.PrivateLayer.Transforms.Type;
        end
        end
        
        function val = get.DataAugmentation(this)
        n = numel(this.PrivateLayer.TrainTransforms);
        if n == 1
            val = this.PrivateLayer.TrainTransforms.Type;
        elseif n > 1
            val = {this.PrivateLayer.TrainTransforms(:).Type};
        else
            val = 'none';
        end
        end
    end
    
    methods(Static)
        function this = loadobj(in)
        internalLayer = nnet.internal.cnn.layer.ImageInput( ...
            in.Name, in.InputSize, ...
            iLoadTransforms( in.Normalization ), ...
            iLoadTransforms( in.Augmentations ) );
        this = nnet.cnn.layer.ImageInputLayer(internalLayer);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
        imageSizeString = i3DSizeToString( this.InputSize );
        
        normalizationString = [ '''' this.Normalization '''' ];
        augmentationsString = iAugmentationsString( this.DataAugmentation );
        
        if strcmp(this.Normalization, 'none') && all(strcmp(this.DataAugmentation, 'none'))
            % No transformations
            description = iGetMessageString( ...
                'nnet_cnn:layer:ImageInputLayer:oneLineDisplayNoTransforms', ....
                imageSizeString );
        elseif strcmp(this.Normalization, 'none')
            % Only augmentations
            description = iGetMessageString( ...
                'nnet_cnn:layer:ImageInputLayer:oneLineDisplayAugmentations', ....
                imageSizeString, ...
                augmentationsString );
        elseif all(strcmp(this.DataAugmentation, 'none'))
            % Only normalization
            description = iGetMessageString( ...
                'nnet_cnn:layer:ImageInputLayer:oneLineDisplayNormalization', ....
                imageSizeString, ...
                normalizationString );
        else
            % Both filled
            description = iGetMessageString( ...
                'nnet_cnn:layer:ImageInputLayer:oneLineDisplay', ....
                imageSizeString, ...
                normalizationString, ...
                augmentationsString );
        end
        
        type = iGetMessageString( 'nnet_cnn:layer:ImageInputLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
        generalParameters = {
            'Name'
            'InputSize'
            };
        
        hyperParameters = {
            'DataAugmentation'
            'Normalization'
            };
        
        groups = [
            this.propertyGroupGeneral( generalParameters )
            this.propertyGroupHyperparameters( hyperParameters )
            ];
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function sizeString = i3DSizeToString( sizeVector )
% i3DSizeToString   Convert a 3-D size stored in a vector of 3 elements
% into a string separated by 'x'.
sizeString = [ ...
    int2str( sizeVector(1) ) ...
    'x' ...
    int2str( sizeVector(2) ) ...
    'x' ...
    int2str( sizeVector(3) ) ];
end

function string = iAugmentationsString( augmentations )
% iAugmentationsString   Convert a cell array of augmentation types into a
% single string. Each augmentation type will be wrapped in '' and separated
% by a coma.
% If augmentations is only one string it will simply return it wrapped in ''.
if iscell( augmentations )
    string = ['''' strjoin( augmentations, ''', ''' ) ''''];
else
    string = ['''' augmentations ''''];
end
end

function S = iSaveTransforms(transforms)
% iSaveTransforms   Save a vector of transformations in the form of an
% array of structures
S = arrayfun( @serialize, transforms );
end

function transforms = iLoadTransforms( S )
% iLoadTransforms   Load a vector of transformations from an array of
% structures S
transforms = nnet.internal.cnn.layer.ImageTransform.empty();
for i=1:numel(S)
    transforms = horzcat(transforms, iLoadTransform( S(i) )); %#ok<AGROW>
end
end

function transform = iLoadTransform(S)
% iLoadTransform   Load a transformation from a structure S
transform = nnet.internal.cnn.layer.ImageTransformFactory.deserialize( S );
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end
