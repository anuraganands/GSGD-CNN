classdef Gradients
    % Gradients   Methods to compute gradients for visualization
    %
    %   Gradients contains methods to compute forward and backward
    %   propagation for an internal VisualNetwork.
    
    %   Copyright 2016 The MathWorks, Inc.
    
    methods(Static)
        
        function [gradient, activations] = computeGradient(...
                iVisualNet, X)
            % Computes the gradient for the visual network at X.
            
            [layerOutputs, memory] = nnet.internal.cnn.visualize.Gradients.forwardPropagation(...
                iVisualNet, X);
            
            [activations, dxLayers] = nnet.internal.cnn.visualize.Gradients.backwardPropagation(...
                iVisualNet, layerOutputs, memory);
            gradient = dxLayers{2};
        end
        
        function [activations, dxLayers] = backwardPropagation(iVisualNet, layerOutputs,...
                memory)
            % Backpropagation through the network.
            
            numLayers = numel(iVisualNet.Layers);
            dxLayers = cell(numLayers, 1);
            activations = layerOutputs{numLayers};
            
            dZ = iVisualNet.Layers{numLayers}.backwardActivations(...
                layerOutputs{numLayers-1}, layerOutputs{numLayers});
            dxLayers{numLayers} = dZ;
                        
            for el = numLayers-1:-1:2
                dxLayers{el} = iVisualNet.Layers{el}.backward(...
                    layerOutputs{el-1}, layerOutputs{el}, dxLayers{el+1}, memory{el});
            end
        end
        
        function [layerOutputs, memory] = forwardPropagation(iVisualNet, X)
            % Forward propagation for X
            
            numLayers = numel(iVisualNet.Layers);
            layerOutputs = cell(numLayers,1);
            memory = cell(numLayers,1);                        
            
            [layerOutputs{1}, memory{1}] = iVisualNet.Layers{1}.forward( X );
            for currentLayer = 2:numLayers-1
                [layerOutputs{currentLayer}, memory{currentLayer}] = iVisualNet.Layers{currentLayer}.forward( layerOutputs{currentLayer-1} );
            end
            layerOutputs{numLayers} = iVisualNet.Layers{numLayers}.forwardActivations( layerOutputs{numLayers-1} );
            memory{numLayers} = [];
        end
    end
end