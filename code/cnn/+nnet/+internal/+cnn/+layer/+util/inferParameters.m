function [layers, analysis] = inferParameters(layers, returnExternal)
    % inferParameters   Infer parameters of an array of layers or a layer
    %                   graph

    %   Copyright 2017 The MathWorks, Inc.

    analysis = nnet.internal.cnn.analyzer.NetworkAnalyzer(layers);
    analysis.applyConstraints();
    analysis.throwIssuesIfAny()
    
    if nargin > 1 && returnExternal == "external"
        layers = analysis.ExternalLayers;
    else
        layers = analysis.InternalLayers;
    end
    
end