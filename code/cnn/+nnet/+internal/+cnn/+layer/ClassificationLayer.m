classdef (Abstract) ClassificationLayer < nnet.internal.cnn.layer.OutputLayer
    % OutputLayer     Interface for convolutional neural network
    % classification output layers
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Dependent)
        % ClassNames (cellstr) The names of the classes
        ClassNames
    end
    
    properties (Abstract)
        % Categories (categorical array) The categories of the classes
        % It can store ordinality of the classes as well.
        Categories
    end
    
    properties (Abstract, SetAccess = private)
        % NumClasses   Number of classes
        NumClasses
    end
    
    methods
        function names = get.ClassNames( this )
            names = categories( this.Categories );
            % make sure this is indeed cellstr
            names = cellstr(names);
        end
    end
end
