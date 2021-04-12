classdef (Abstract) SummaryUpdater < handle
    % SummaryUpdater   Interface for classes that can update
    % MiniBatchSummary objects
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods (Abstract)
        % updateSummary   Compute predictions on the validation set using
        % the network net according to the current iteration and update the
        % MiniBatchSummary summary
        updateSummary(this, summary, net)
    end
end

