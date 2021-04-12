classdef ExternalInternalConverter
    % ExternalInternalConverter   Class to convert an external layer into
    % an internal one
    
    %   Copyright 2017 The MathWorks, Inc.
    methods (Static)
        function internalLayers = getInternalLayers(layers)
            % getInternalLayers   Return the corresponding internal layers
            % as a cell-array of dimension n-by-1, where n is the number of
            % layers
            internalLayers = cell(numel(layers),1);
            for ii = 1:numel(layers)
                if isa(layers(ii), 'nnet.internal.cnn.layer.Externalizable')
                    internalLayers{ii} = layers(ii).PrivateLayer;
                elseif isa(layers(ii), 'nnet.layer.ClassificationLayer')
                    internalLayers{ii} = nnet.internal.cnn.layer.CustomClassificationLayer( layers(ii), iLayerVerifier() );
                elseif isa(layers(ii), 'nnet.layer.RegressionLayer')
                    internalLayers{ii} = nnet.internal.cnn.layer.CustomRegressionLayer( layers(ii), iLayerVerifier() );
                else
                    internalLayers{ii} = nnet.internal.cnn.layer.CustomLayer( layers(ii), iLayerVerifier() );
                end
            end
        end
    end
end

function layerValidator = iLayerVerifier()
layerValidator = nnet.internal.cnn.layer.util.CustomLayerVerifier();
end