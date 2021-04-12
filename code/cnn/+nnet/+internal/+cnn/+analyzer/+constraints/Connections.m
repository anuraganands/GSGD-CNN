classdef Connections < nnet.internal.cnn.analyzer.constraints.Constraint
    % Connections   Constrain object to be used by analyzeNetwork.
    %               Detects issues related to the connection between layers
    %               int the network.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        
        function testConnectionsToInvalidPorts(test)
            % Connections to invalid ports can happen when we create a
            % series network. In this case we can create a connection:
            %  * from a layer with no output.
            %  * to a layer with no input.
            %
            % A layer with no input should be the first layer in a network
            % (we expect this to be an input layer).
            % A layer with no output should be the last layer in a network
            % (we expect this to be an output layer).
            
            nInputLayers = sum([test.LayerAnalyzers.IsInputLayer]);
            nOutputLayers = sum([test.LayerAnalyzers.IsOutputLayer]);
            
            for i=1:size(test.InternalConnections, 1)
                conn = test.InternalConnections(i,:);
                src = conn(1);
                dst = conn(3);
                
                if test.LayerAnalyzers(dst).IsInputLayer && nInputLayers > 1
                    % Nothing to do here, this is handled by 
                    % Architecture/testOneInputLayer
                elseif test.LayerAnalyzers(dst).IsInputLayer
                    test.addLayerError(dst, ...
                        "Connections:ConnectionsToInputLayer" );
                elseif isempty(test.LayerAnalyzers(dst).Inputs)
                    test.addLayerError(dst, ...
                        "Connections:ConnectionsToInvalidPort", ...
                        test.sourceNames( conn(1), conn(2) ) );
                end
                
                if test.LayerAnalyzers(src).IsOutputLayer && nOutputLayers > 1
                    % Nothing to do here, this is handled by 
                    % Architecture/testOneOutputLayer
                elseif test.LayerAnalyzers(src).IsOutputLayer
                    test.addLayerError(src, ...
                        "Connections:ConnectionsFromOutputLayer");
                elseif isempty(test.LayerAnalyzers(src).Outputs)
                    test.addLayerError(src, ...
                        "Connections:ConnectionsFromInvalidPort", ...
                        test.destinationNames( conn(3), conn(4) ) );
                end
            end
        end
        
        function testConnectionCycles(test)
            % When the network to analyze comes from a layerGraph, the
            % network to be analyzed can include cycles in the connections.
            %
            % In a topologically sorted acyclic network the source index
            % is strictly lower than the destination index.
            % The network analyzer produces a pseudo topologically sorted
            % list of layers, so any connection with a source index higher
            % to the destination index indicates that the connection would
            % create a cycle (equal indexes indicate a self loop).
            
            src = test.InternalConnections(:,1);
            dst = test.InternalConnections(:,3);
            
            for i=find(src >= dst)'
                conn = test.InternalConnections(i,:);
                test.addIssue("E", "Network", [src(i) dst(i)], ...
                    "Connections:ConnectionCycle", ...
                    test.sourceNames(conn(1), conn(2)), ...
                    test.destinationNames(conn(3), conn(4)));
            end
        end
        
        function testMissingConnections(test)
            % Test whether a layer is missing a connection to any of its
            % inputs or has unused outputs.
            
            for i=1:numel(test.LayerAnalyzers)
                missingInputs = iIsMissing(test.LayerAnalyzers(i).Inputs);
                missingOutputs = iIsMissing(test.LayerAnalyzers(i).Outputs);
                
                if all(missingInputs) && all(missingOutputs)
                    % This is a disconnected layer, this issue is captured
                    % by the architecture check "testConnectedComponents".
                    continue;
                end
                
                missingInputs = test.LayerAnalyzers(i).Inputs.Port(missingInputs);
                if isempty(missingInputs)
                    % Nothing to do here, everything is alright
                elseif size(test.LayerAnalyzers(i).Inputs,1) == 1
                    % Only one possible input, and it is missing.
                    % Add a message without naming the input.
                    test.addLayerError(i, ...
                        "Connections:MissingInputs" );
                else
                    % Add a message without naming the missing input(s).
                    test.addLayerError(i, ...
                        "Connections:MissingInputs", ...
                        missingInputs );
                end
                
                missingOutputs = test.LayerAnalyzers(i).Outputs.Port(missingOutputs);
                if isempty(missingOutputs)
                    % Nothing to do here, everything is alright
                elseif size(test.LayerAnalyzers(i).Outputs,1) == 1
                    % Only one possible output, and it is missing.
                    % Add a message without naming the output.
                    test.addLayerError(i, ...
                        "Connections:UnusedOutputs" );
                else
                    % Add a message without naming the missing output(s).
                    test.addLayerError(i, ...
                        "Connections:UnusedOutputs", ...
                        missingOutputs );
                end
            end
        end
        
    end
    
    methods
        function srcName = sourceNames(test, layers, ports)
            % Converts a layer index and output port index to "layer XX
            % output YY" or "layer XX" if there's only one output port.
            
            srcName = strings(size(layers));
            for i=1:numel(layers)
                la = test.LayerAnalyzers(layers(i));

                if isscalar(la.Outputs.Port)
                    srcName(i,1) = string(iMessage( ...
                        'Connections:OutputWithoutPort', ...
                        la.DisplayName));
                else
                    srcName(i,1) = string(iMessage( ...
                        'Connections:OutputWithPort', ...
                        la.DisplayName, la.Outputs.Port{ports(i)}));
                end
            end
        end
        function srcName = destinationNames(test, layers, ports)
            % Converts a layer index and input port index to "layer XX
            % input YY" or "layer XX" if there's only one input port.
            
            srcName = strings(size(layers));
            for i=1:numel(layers)
                la = test.LayerAnalyzers(layers(i));

                if isscalar(la.Inputs.Port)
                    srcName(i,1) = string(iMessage( ...
                        'Connections:InputWithoutPort', ...
                        la.DisplayName));
                else
                    srcName(i,1) = string(iMessage( ...
                        'Connections:InputWithPort', ...
                        la.DisplayName, la.Inputs.Port{ports(i)}));
                end
            end
        end
    end
end

function missing = iIsMissing(tbl, column)
    if nargin < 2
        if any(tbl.Properties.VariableNames == "Source")
            column = 'Source';
        elseif any(tbl.Properties.VariableNames == "Destination")
            column = 'Destination';
        end
    end
    missing = cellfun(@isempty, tbl.(column));
end

function msg = iMessage(id, varargin)
id = "nnet_cnn:internal:cnn:analyzer:constraints:" + id;
msg = message(char(id), varargin{:});
end