classdef Propagation < nnet.internal.cnn.analyzer.constraints.Constraint
    % Propagation   Constrain object to be used by analyzeNetwork.
    %               Detects issues related to propagating sizes through
    %               the network.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        
        function testInvalidLayerSize(test)
            % When propagating sizes through the layer we might find
            % situations (i.e., a missing input) that prevent us from
            % obtaining the output size. In that case, the output will
            % contain NaN values.
            %
            % In some layers, we can obtain a more detailed error message
            % by calling the isValidInputSize method of their internal
            % layer, which may throw a more useful message.
            %
            % An invalid output size in one layer might prevent us from
            % continuing propagating sizes in following layers, in that
            % case we should only report an error in the layer that
            % originated the problem.
            
            badSize = arrayfun(@iHasInvalidOutputSize, test.LayerAnalyzers);
            
            for i=find(badSize')
                inputSizes = test.LayerAnalyzers(i).Inputs.Size;
                inputSizes = iUnwrapScalarSizeCell(inputSizes);
                
                if iAnyIsInvalidSizes(inputSizes)
                    % This layer has a bad size because of one of its
                    % inputs, don't report it as a new error.
                    continue;
                end
                
                % Build a generic input size mismatch message
                msg = test.inputSizeMismatchGenericMessage(i);
                
                % Check if we can get a custom error message
                try
                    test.LayerAnalyzers(i)...
                        .InternalLayer.inferSize(inputSizes);
                    test.LayerAnalyzers(i)...
                        .InternalLayer.isValidInputSize(inputSizes);
                catch err
                    % Check is the error is something we have thrown, which
                    % might contain interesting diagnostics and use that
                    % instead.
                    if startsWith(err.identifier, "nnet_cnn:internal:cnn:layer:CustomLayer")
                        % This is handled by the custom layer constraint.
                        msg = {};
                    elseif startsWith(err.identifier, "nnet_cnn:")
                        msg = {err.identifier ...
                            "Propagation:InvalidLayerSizeWithCause" ...
                            err.message};
                    else
                        % It is not our error message, so use the generic
                        % diagnostic.
                    end
                end
                
                % Now add the error message to the layer (if we didn't
                % decide that the error is handled by someone else.
                if ~isempty(msg)
                    test.addLayerErrorWithId(i, msg{:});
                end
            end
        end
        
    end
    
    methods
        
        function msg = inputSizeMismatchGenericMessage(test, layerIndex)
            % Build the message argument with the name and size of
            % the input ports
            
            la = test.LayerAnalyzers(layerIndex);
            
            conn = test.InternalConnections;
            conn = conn(conn(:,3) == layerIndex, :);
            conn = sortrows(conn, 4);
            
            inputNames = test.sourceNames(conn(:,1), conn(:,2));
            inputSizes = iSizesToString(la.Inputs.Size(conn(:,4)));

            messageArgs = [inputNames(:) inputSizes(:)];
            msg = {"" "Propagation:InvalidLayerSize", messageArgs};
        end
        
        function srcName = sourceNames(test, layers, ports)
            % Converts a layer index and output port index to "layer XX
            % output YY" or "layer XX" if there's only one output port.
            
            srcName = strings(size(layers));
            for i=1:numel(layers)
                la = test.LayerAnalyzers(layers(i));

                if isscalar(la.Outputs.Port)
                    srcName(i,1) = string(iMessage( ...
                        'Propagation:OutputWithoutPort', ...
                        la.DisplayName));
                else
                    srcName(i,1) = string(iMessage( ...
                        'Propagation:OutputWithPort', ...
                        la.DisplayName, la.Outputs.Port{ports(i)}));
                end
            end
        end
        
    end
end

function n = countRows(in)
    n = size(in,1);
end

function sz = iUnwrapScalarSizeCell(sz)
    if isscalar(sz)
        sz = sz{1};
    end
end

function tf = iHasInvalidOutputSize(layerAnalyzer)
    tf = iAnyIsInvalidSizes(layerAnalyzer.Outputs.Size);
end

function tf = iAnyIsInvalidSizes(sizes)
    tf = any(~iIsValidSizes(sizes));
end

function tf = iIsValidSizes(sz)
    % Iterate over a cell array of input sizes and returns a logic value
    % for each element in the cell, indicating whether that element is a
    % valid size vector or not.
    
    if ~iscell(sz)
        sz = {sz};
    end
    tf = false(size(sz));
    for i=1:numel(sz)
        tf(i) = ~isempty(sz{i}) ...
                && all(iIsNatural(sz{i}));
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

function str = iSizesToString(sz)
str = cellfun(@iSizeToString, sz);
end

function str = iSizeToString(sz)
str = join(string(sz), matlab.internal.display.getDimensionSpecifier);
if isscalar(sz)
    str = string(iMessage('Propagation:ScalarOutputSize', str));
else
    str = string(iMessage('Propagation:VectorOutputSize', str));
end
end

function msg = iMessage(id, varargin)
id = "nnet_cnn:internal:cnn:analyzer:constraints:" + id;
msg = message(char(id), varargin{:});
end