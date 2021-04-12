classdef ImagePerformanceStrategy < nnet.internal.cnn.util.PerformanceStrategy
    % ImagePerformanceStrategy   Summary performance execution strategy for image data
    %
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function acc = accuracy(~, predictions, response)
            % accuracy   Classification accuracy. predictions and response
            % are of size height-by-width-by-numClasses-by-numObservations
            [~, yIntegers] = max(predictions, [], 3);
            [~, tIntegers] = max(response, [], 3);
            numObservations = size(yIntegers,1) * size(yIntegers,2) * size(yIntegers, 4);
            acc = 100 * ( sum(sum(sum( yIntegers == tIntegers, 1 ), 2), 4)/ numObservations );
        end
        
        function err = rmse(~, predictions, response)
            % rmse   Root-mean-squared-error. predictions and response are
            % of size height-by-width-by-outputSize-by-numObservations
            squares = (predictions-response).^2;
            numObservations = size( squares, 4 );
            err = sqrt( sum( squares (:) ) / numObservations );
        end
    end
end