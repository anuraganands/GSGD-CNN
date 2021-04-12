classdef PiecewiseSchedule < nnet.internal.cnn.LearnRateSchedule
    % PiecewiseSchedule   An object representing a piecewise constant
    % learning rate schedule
    
    %   Copyright 2015 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % DropFactor   Multiplicative factor for dropping the learning rate
        %   A multiplicative factor that is applied to the learning rate
        %   every time a certain number of iterations is passed. The number
        %   of iterations that must pass is defined by DropPeriod.
        DropFactor
        
        % DropPeriod   Number of epochs that must pass before dropping the learning rate
        %   The number of epochs between applying DropFactor to the
        %   learning rate.
        DropPeriod
    end
    
    methods
        function this = PiecewiseSchedule(dropFactor, dropPeriod)
            this.DropFactor = dropFactor;
            this.DropPeriod = dropPeriod;
        end
        
        function newLearnRate = update(this, oldLearnRate, epoch)
            if(iNeedToDropLearningRate(epoch, this.DropPeriod))
                newLearnRate = this.DropFactor*oldLearnRate;
            else
                newLearnRate = oldLearnRate;
            end
        end
    end
end

function tf = iNeedToDropLearningRate(epoch, dropPeriod)
tf = rem( epoch, dropPeriod ) == 0;
end