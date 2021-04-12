classdef ExecutionInfo < handle
    % ExecutionInfo   Holds fixed information about how training is executed
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % ExecutionEnvironment   (char) One of 'cpu' or 'gpu'
        ExecutionEnvironment
        
        % UseParallel   (logical) 
        UseParallel
        
        % LearnRateSchedule   (char) One of 'piecewise' or 'constant'
        LearnRateSchedule
        
        % InitialLearningRate   (double) Initial learning rate
        InitialLearningRate
    end
    
    methods
        function this = ExecutionInfo(executionEnvironment, useParallel, learnRateSchedule, initialLearningRate)
            this.ExecutionEnvironment = executionEnvironment;
            this.UseParallel = useParallel;
            this.LearnRateSchedule = learnRateSchedule;
            this.InitialLearningRate = initialLearningRate;
        end
    end
    
end

