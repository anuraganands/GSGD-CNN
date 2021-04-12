classdef VectorPerformanceStrategy < nnet.internal.cnn.util.PerformanceStrategy
    % VectorPerformanceStrategy   Summary performance execution strategy for vector data
    %
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function acc = accuracy(~, predictions, response)
            % accuracy   Classification accuracy. predictions and response
            % are of size numClasses-by-numObservations-by-sequenceLength
            [~, yIntegers] = max(predictions, [], 1);
            [~, tIntegers] = max(response, [], 1);
            totalElems = size(yIntegers, 2)*size(yIntegers, 3);
            acc = 100 * ( sum( yIntegers(:) == tIntegers(:) )/ totalElems );
        end
        
        function err = rmse(~, predictions, response)
            % rmse   Root-mean-squared-error. predictions and response are
            % of size outputSize-by-numObservations-by-sequenceLength
            squares = (predictions-response).^2;
            totalElems = size( squares, 2 )*size( squares, 3 );
            err = mean( sqrt( sum( squares (:) ) / totalElems ) );
        end
    end
end