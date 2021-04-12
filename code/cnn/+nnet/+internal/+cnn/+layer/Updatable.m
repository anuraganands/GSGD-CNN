classdef (Abstract) Updatable
    % Updatable   Mixin for layers that support state update
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods (Abstract)
        % computeState  Use the inputs and outputs from forward to
        %                   compute state information for this layer
        %
        % Syntax
        %   state = computeState(layer, X, Z, memory, propagateState)
        %
        % Inputs
        %   layer          - a layer
        %   X              - the input to forward for the layer
        %   Z, memory      - the outputs from the layer's forward method
        %   propagateState - true to propagate previous state and false to
        %                    re-initialize the state
        %
        % Outputs
        %   state          - a cell array containing state information
        state = computeState(layer, X, Z, memory, propagateState)
        
        % updateState  Update state information for this layer
        %
        % Syntax
        %   layer = updateState(layer, state)
        %
        % Inputs
        %   state       - a cell array containing state information
        %                 from a previous call to computeState
        %
        % Outputs
        %   layer       - the updated layer
        layer = updateState(layer, state)
    end
end