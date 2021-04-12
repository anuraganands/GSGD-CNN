classdef Architecture < nnet.internal.cnn.analyzer.constraints.Constraint
    % Architecture  Constraint object to be used by analyzeNetwork.
    %               Detects issues related to the architecture of the
    %               network.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        
        function testOneInputLayer(test)
            % Test that the network has only one input layer.
            %
            % At least one input layer is always a valid constraint for
            % both, DAG and series networks.
            % No more than one input layer is a always valid constrain for
            % series networks, and is a valid constraint for our current
            % DAG implementation until we add support for multiple inputs.
            
            isInput = [test.LayerAnalyzers.IsInputLayer];
            names = [test.LayerAnalyzers(isInput).Name]';
            
            if isempty(names)
                test.addIssue("E", "Network", [], ...
                    "Architecture:MissingInputLayer");
            elseif ~isscalar(names)
                test.addIssue("E", "Network", names, ...
                    "Architecture:OneInputLayer", ...
                    [test.LayerAnalyzers(names).DisplayName]');
            end
        end
        
        function testOneOutputLayer(test)
            % Test that the network has only one output layer.
            %
            % At least one output layer is always a valid constraint for
            % both, DAG and series networks.
            % No more than one output layer is a always valid constrain for
            % series networks, and is a valid constraint for our current
            % DAG implementation until we add support for multiple outputs.
            
            isOutput = [test.LayerAnalyzers.IsOutputLayer];
            names = [test.LayerAnalyzers(isOutput).Name]';
            
            if isempty(names)
                test.addIssue("E", "Network", [], ...
                    "Architecture:MissingOutputLayer");
            elseif ~isscalar(names)
                test.addIssue("E", "Network", names, ...
                    "Architecture:OneOutputLayer", ...
                    [test.LayerAnalyzers(names).DisplayName]');
            end
        end
        
        function testConnectedComponents(test)
            % Test that the network consists of one connected component.
            
            % Construct a digraph and obtain the connected components
            conn = test.NetworkAnalyzer.HiddenConnections.EndNodes;
            n = numel(test.LayerAnalyzers);
            g = digraph(conn(:,1),conn(:,2),[],n);
            bins = conncomp(g, 'Type', 'weak');

            if max(bins) == 1
                % The graph just have one component. Everything is alright.
                return;
            end
            
            % Obtain the first element of each component, and the number of
            % layers in that component.
            [~, iComp] = unique(bins, 'stable');
            nComp = hist(bins, 1:max(bins));
            
            % Sort the components by number of layers
            [nComp, i] = sort(nComp(:), 'descend');
            iComp = iComp(i);

            % Get the leading layers names
            names = [test.LayerAnalyzers(iComp).Name]';
            displayName = [test.LayerAnalyzers(iComp).DisplayName]';
            
            % Divide in componens and isolated layers
            isIsolated = ( nComp == 1 );
            
            isolated.names = names(isIsolated);
            isolated.displayName = displayName(isIsolated);
            
            nonisolated.names = names(~isIsolated);
            nonisolated.displayName = displayName(~isIsolated);
            nonisolated.size = string(nComp(~isIsolated));
            
            if numel(nonisolated.names) > 1
                test.addIssue("E", "Network", nonisolated.names, ...
                    "Architecture:MultipleComponents", ...
                    [nonisolated.displayName, nonisolated.size]);
            end
            
            if numel(isolated.names)
                test.addIssue("E", "Network", isolated.names, ...
                    "Architecture:DisconnectedLayers", ...
                    isolated.displayName);
            end
            
            %{
            % Report an error on all the other components, if any.
            for i=iComp(:)'
                test.addLayerError(i, ...
                    "Architecture:MultipleComponents");
            end
            %}
            
        end
        
        function testClassificationMustBePrecededBySoftmax(test)
            % Test that all classification layers are preceded by a softmax
            % layer.
            %
            % The input to a classification layer has to be a probability
            % vector, which is what a softmax layer does.
            
            src = test.InternalConnections(:,1);
            dst = test.InternalConnections(:,3);
            
            for i=1:numel(test.LayerAnalyzers)
                if ~test.LayerAnalyzers(i).IsClassficationLayer
                    continue;
                end
                
                sources = src(dst == i);
                srcSoftmax = [test.LayerAnalyzers(sources).IsSoftmaxLayer];
                
                offending = sources(~srcSoftmax);
                offending = {test.LayerAnalyzers(offending).Name}';
                
                if ~isempty(offending)
                    test.addLayerError(i, ...
                        "Architecture:ClassificationMustBePrecededBySoftmax" );
                end
            end
        end
        
        function testRegressionMustNotBePrecededBySoftmax(test)
            % Test that all regression layers are not preceded by a softmax
            % layer.
            %
            % The input to a regression layer should be able to take any
            % arbitrary values. A softmax layer produces probability
            % vectors, so they are not suitable as the output of a
            % regression.
            
            src = test.InternalConnections(:,1);
            dst = test.InternalConnections(:,3);
            
            for i=1:numel(test.LayerAnalyzers)
                if ~test.LayerAnalyzers(i).IsRegressionLayer
                    continue;
                end
                
                sources = src(dst == i);
                srcSoftmax = [test.LayerAnalyzers(sources).IsSoftmaxLayer];
                
                offending = sources(srcSoftmax);
                offending = {test.LayerAnalyzers(offending).Name}';
                
                if ~isempty(offending)
                    test.addLayerError(i, ...
                        "Architecture:RegressionMustNotBePrecededBySoftmax" );
                end
            end
        end
        
    end
end
