classdef InternalExternalMap
    % InternalExternalMap   Class that holds the internal to external
    % layers map
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties (Access = private)
        % LayersMap   (containers.Map) Map that holds the internal layers
        %                              class names as keys and the external
        %                              layers constructors as corresponding
        %                              values.
        LayersMap = containers.Map.empty;
    end
    
    methods
        function this = InternalExternalMap( externalLayers )
            this.LayersMap = iLayersMap( externalLayers );
        end
        
        function externalLayers = externalLayers( this, internalLayers )
            % externalLayers   Take as input a cell-array of internal
            % layers and return the corresponding external layers, as an
            % n-by-1 heterogeneous array, where n is the number of layers.
            internalLayersClasses = iInternalLayersClasses( internalLayers );
            externalLayersConstructors = this.LayersMap.values( internalLayersClasses );
            externalLayers = iConstructExternalLayers( externalLayersConstructors, internalLayers );
        end
    end
end

function layersMap = iLayersMap( externalLayers )
internalLayers = iInternalLayers( externalLayers );
keys = iInternalLayersClasses( internalLayers );
values = iLayersConstructors( externalLayers );
layersMap = containers.Map( keys, values );
end

function layersConstructors = iLayersConstructors( layers )
layersConstructors = arrayfun( @iLayerConstructor, layers, 'UniformOutput', false );
end

function layerConstructor = iLayerConstructor( externalLayer )
% iLayerConstructor    Function handle to constructor for an external layer
%
% Signature of the constructor is
%      extL = FCN( intL )
% where intL is an internal layer

if isa(externalLayer, 'nnet.internal.cnn.layer.Externalizable')
    % For built-in layers, derive the constructor from the class name
    layerConstructor = str2func(class(externalLayer));
else
    % For a custom layer, just extract the "external layer" from the "internal layer"
    layerConstructor = @iGetExternalCustomLayerFromInternal;    
end
end

function externalLayer = iGetExternalCustomLayerFromInternal(internalLayer)
externalLayer = internalLayer.ExternalCustomLayer;
end

function internalLayers = iInternalLayers( externalLayers )
internalLayers = nnet.cnn.layer.Layer.getInternalLayers( externalLayers );
end

function layersClasses = iInternalLayersClasses( internalLayers )
layersClasses = cellfun( @class, internalLayers, 'UniformOutput', false );
end

function externalLayers = iConstructExternalLayers( externalLayersConstructors, internalLayers )
numLayers = numel( externalLayersConstructors );
for ii=1:numLayers
    externalLayers(ii,1) = externalLayersConstructors{ii}(internalLayers{ii});
end
end
