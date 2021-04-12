classdef LSTM < nnet.internal.cnn.analyzer.constraints.Constraint
    % LSTM  Constrain object to be used by analyzeNetwork.
    %       Detects issues related to LSTM networks.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        
        function testRecurrentAndImageLayers(test)
            % Test that image specific layers and recurrent layers
            % are not combined in the same network.

            recurrent = [test.LayerAnalyzers.IsRNNLayer];
            image     = [test.LayerAnalyzers.IsImageSpecificLayer];
            
            rnnLayers = {test.LayerAnalyzers(recurrent).DisplayName
                         test.LayerAnalyzers(recurrent).Type}';
            imgLayers = {test.LayerAnalyzers(image).DisplayName
                         test.LayerAnalyzers(image).Type}';

            if any(recurrent) && any(image)
                test.addIssue("E", "Network", find(recurrent | image), ...
                    "LSTM:RecurrentAndImageLayers", imgLayers, rnnLayers);
            end
        end
        
        function testSequenceInputAndImageLayers(test)
            % Test that image specific layers and sequence input layers
            % are not combined in the same network.

            sequence  = [test.LayerAnalyzers.IsSequenceSpecificLayer];
            recurrent = [test.LayerAnalyzers.IsRNNLayer];
            image     = [test.LayerAnalyzers.IsImageSpecificLayer];

            if any(recurrent)
                % A message is already shown by the test
                % "testRecurrentAndImageLayer", don not show a new one
                return;
            end

            seqInput = ( sequence & ~recurrent );
            imgLayers = {test.LayerAnalyzers(image).DisplayName
                         test.LayerAnalyzers(image).Type}';

            if any(seqInput) && any(image)
                test.addIssue("E", "Network", find(seqInput | image), ...
                    "LSTM:SequenceInputAndImageLayers", imgLayers);
            end
        end

    end
end