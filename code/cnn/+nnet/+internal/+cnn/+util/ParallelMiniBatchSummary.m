classdef ParallelMiniBatchSummary < nnet.internal.cnn.util.MiniBatchSummary
    % ParallelMiniBatchSummary   Extends MiniBatchSummary to coordinate
    % updates between parallel workers
    
    %   Copyright 2016 The MathWorks, Inc.
    
    methods
        function this = ParallelMiniBatchSummary(varargin)
            this = this@nnet.internal.cnn.util.MiniBatchSummary(varargin{:});
        end
        function update( this, predictions, response, epoch, iteration, elapsedTime, miniBatchLoss, learnRate, lossFunctionType )
            % update  Overload to merge output between workers
            update@nnet.internal.cnn.util.MiniBatchSummary(this, predictions, response, epoch, iteration, elapsedTime, miniBatchLoss, learnRate);
            merge(this, lossFunctionType);
        end
    end
    
    methods( Access = private )
        function merge( this, lossFunctionType )
            % merge   Only call within SPMD block, to merge current Loss
            % and Accuracy between workers. Note that predictions and
            % response are NOT merged because this isn't needed.
            subBatchSize = size(this.Predictions, 4);
            mergedLossAndAccuracy = gop(@iBlendLossAndAccuracy, ...
                { this.Loss, this.Accuracy, subBatchSize, lossFunctionType } );
            this.Loss = mergedLossAndAccuracy{1};
            this.Accuracy = mergedLossAndAccuracy{2};
        end
    end
end

function [val, n] = iWeightedBlend(val1, n1, val2, n2)
% Blend two values with appropriate weights
n = n1 + n2;
if n == 0
    val = 0;
else
    val = (n1*val1/n) + (n2*val2/n);
end
end

function blended = iBlendLossAndAccuracy(args1, args2)
% Blend the loss and accuracy. Function in binary form for use by GOP

% Retrieve individual arguments
[loss1, accuracy1, n1, outputType1] = deal( args1{:} );
[loss2, accuracy2, n2, ~] = deal( args2{:} );

% Blend accuracy
[accuracy, n] = iWeightedBlend( accuracy1, n1, accuracy2, n2 );

% Blend loss based on loss function type
switch ( outputType1 )
    otherwise
        % Default which works for most loss types
        [loss, ~] = iWeightedBlend( loss1, n1, loss2, n2 );
end

% Combine into a single output
blended = {loss, accuracy, n, outputType1};
end