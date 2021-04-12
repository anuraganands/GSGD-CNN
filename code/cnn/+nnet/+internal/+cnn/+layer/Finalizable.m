classdef (Abstract) Finalizable
    % Finalizable  Mixin for layers that require a finalization pass at
    % the end of training.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods (Abstract)
        % finalize  Use the inputs and outputs from forward to finalize this layer
        %
        % Syntax
        %   layer = finalize(layer, X, Y, memory)
        %
        % Inputs
        %   layer     - the layer to be finalized
        %   X         - the input to forward for the layer
        %   Y, memory - the outputs from the layer's forward method
        %
        % Outputs
        %   layer     - the updated layer
        layer = finalize(layer, X, Y, memory)
        
        % mergeFinalized  Combine two partially finalized layer objects.
        %
        % When finalizing in parallel, each worker will build up a
        % finalized copy of the layer. These must then be merged together
        % to give one final result.
        %
        % Syntax
        %   layer = mergeFinalized(layer1, layer2)
        %
        % Inputs
        %   layer1, layer2 - the layers to merge
        %
        % Outputs
        %   layer          - the final merged layer
        layer = mergeFinalized(layer1, layer2)
    end
end
