classdef VisualNetwork
    % VisualNetwork   A class containing methods to create a visual network
    % from an internal SeriesNetwork and a layer and channels.
    %
    %   Gradients contains methods to compute forward and backward
    %   propagation for an internal VisualNetwork.
    
    %   Copyright 2016 The MathWorks, Inc.
    
    methods(Static)
        function iVisualNet = createVisualNetworkForChannelAverage(iNet, layer, channels)
            % Creates a visual network, by adding on an OptimizeChannelAverage
            % layer for the specified channels.

            layers = iNet.Layers(1:layer);
            visualLayers = layers;
            visualLayers{layer+1} = nnet.internal.cnn.visualize.layer.OptimizeChannelAverage(...
                channels);
            iVisualNet = nnet.internal.cnn.SeriesNetwork(visualLayers);
        end
        
        function iVisualNet = createVisualNetworkForDeepDream(iNet, layer)
            % Creates a visual network, by adding on a DeepDream layer

            layers = iNet.Layers(1:layer);
            visualLayers = layers;
            visualLayers{layer+1} = nnet.internal.cnn.visualize.layer.DeepDream();
            iVisualNet = nnet.internal.cnn.SeriesNetwork(visualLayers);
        end
    end
end