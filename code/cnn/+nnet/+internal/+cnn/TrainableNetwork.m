classdef (Abstract) TrainableNetwork
    % TrainableNetwork   Interface for trainable convolutional neural networks
    
    %   Copyright 2015 The MathWorks, Inc.    
    
    properties (Abstract)
        % Layers    Dependent property that lets implement the logic behind
        % set.Layers and get.Layers on the network
        Layers
    end 
    
    methods (Abstract)
        % predict   Predict a response based on data
        response = predict(this, data);      
    end
    
end
