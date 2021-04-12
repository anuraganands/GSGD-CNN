classdef CustomLayers < nnet.internal.cnn.analyzer.constraints.Constraint
    % Names         Constrain object to be used by analyzeNetwork.
    %               Detects issues related to custom layers in the network.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        
        function testValidateLayer(test)
            % Test whether custom layers is valid
            
            for i = find([test.LayerAnalyzers.IsCustomLayer])
                % Validate the method signature
                try
                    iValidateMethodSignatures(...
                        test.LayerAnalyzers(i).ExternalLayer, i );
                catch err
                    test.addLayerErrorWithId(i, err.identifier, ...
                        "CustomLayers:BadMethodSignature", err.message);
                    
                    % Don't continue validating after this, the layer is
                    % already invalid.
                    continue;
                end
                
                % Validate the methods behavior.
                try
                    inputSizes = iGetValidUnwrappedInputSizesOrThrow(...
                        test.LayerAnalyzers(i).Inputs );
                catch
                    % This layer has an invalid input. Trying to check the
                    % input size will error because of this, and not
                    % because of the custom layer.
                    continue;
                end
                
                % Validate the behavior on a single image
                try
                    test.LayerAnalyzers(i)...
                        .InternalLayer.isValidInputSize(inputSizes);
                catch err
                    test.addLayerErrorWithId(i, err.identifier, ...
                        "CustomLayers:CustomLayerVerificationFailed", ...
                        err.message, struct('cause', {err.cause}));
                    
                    % This aready failed, don't validate on minibatch.
                    continue;
                end
                
                % Validate the behavior on a minibatch. This test assumes
                % that there is only one inputSize (i.e., no multi input
                % custom layers)
                if any([test.LayerAnalyzers.IsSequenceSpecificLayer])
                    inputSizes(2) = 5;
                else
                    inputSizes(4) = 5;
                end
                try
                    test.LayerAnalyzers(i)...
                        .InternalLayer.isValidInputSize(inputSizes);
                catch err
                    test.addLayerErrorWithId(i, err.identifier, ...
                        "CustomLayers:CustomLayerVerificationFailed", ...
                        err.message, struct('cause', {err.cause}));
                end
            end
        end
        
    end
end

function iValidateMethodSignatures(layer, ind)
    nnet.internal.cnn.layer.util.CustomLayerVerifier...
        .validateMethodSignatures(layer, ind);
end

function sz = iGetValidUnwrappedInputSizesOrThrow(inputs)
    sz = inputs.Size;
    if isscalar(sz)
        sz = sz{1};
    end

    if any(~iIsValidSize(sz))
        error('dummy:id', 'A dummy error.')
    end
end

function tf = iIsValidSize(outputSizes)
    % Iterate over a cell array of input sizes and returns a logic value
    % for each element in the cell, indicating whether that element is a
    % valid size vector or not.
    
    if ~iscell(outputSizes)
        outputSizes = {outputSizes};
    end
    tf = false(size(outputSizes));
    for i=1:numel(outputSizes)
        tf(i) = ~isempty(outputSizes{i}) ...
                && all(iIsNatural(outputSizes{i}));
    end
end

function tf = iIsNatural(v)
    % Returns true if a number belongs to the group of natural numbers
    try
        validateattributes(v, {'numeric'},{'positive','integer'});
        tf = true;
    catch
        tf = false;
    end
end