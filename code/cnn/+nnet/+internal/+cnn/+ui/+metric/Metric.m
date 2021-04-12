classdef Metric < nnet.internal.cnn.ui.metric.UpdateableMetric
    % Metric   Class which reads information struct and updates its UpdateableSeries if necessary.
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties(Access = private)
        % UpdateableSeries   (nnet.internal.cnn.ui.axes.UpdateableSeries)
        % Series to update 
        UpdateableSeries
        
        % MetricName   (char) Name of struct field name to extract
        MetricName
        
        % CanUpdateAfterTraining  (bool) True if this Metric can have its
        % values updated after the training loop has been completed (e.g.
        % on the finish() step of training).
        CanUpdateAfterTraining
    end
    
    methods
        function this = Metric(updateableSeries, metricName, canUpdateAfterTraining)
            this.UpdateableSeries = updateableSeries;
            this.MetricName = metricName;
            this.CanUpdateAfterTraining = canUpdateAfterTraining;
        end
        
        function update(this, infoStruct)
            % Updates the data in this metric during training.
            
            isDuringTraining = true;
            this.doUpdate(infoStruct, isDuringTraining);
        end
        
        function updatePostTraining(this, infoStruct)
            % Updates the data in this metric after training (e.g. in the
            % Reporter's finish() step, or the post-batch-normalization
            % calculation of validation).
            
            isDuringTraining = false;
            this.doUpdate(infoStruct, isDuringTraining);
        end
    end
    
    methods(Access = private)
       
        function doUpdate(this, infoStruct, isDuringTraining)
            % Performs an update, adding the new metric values to the
            % UpdateableSeries.
            
            metricValue = infoStruct.(this.MetricName);
            if iMetricIsNonEmpty(metricValue) && iMetricCanUpdate(isDuringTraining, this.CanUpdateAfterTraining)
                xValue = infoStruct.Iteration;
                yValue = infoStruct.(this.MetricName);
                this.UpdateableSeries.add(xValue, yValue);
            end
        end
        
    end
end

function tf = iMetricIsNonEmpty(metricValue)
tf = ~isempty(metricValue);
end

function tf = iMetricCanUpdate(isDuringTraining, canUpdateAfterTraining)
tf = (isDuringTraining ||...
    ~isDuringTraining && canUpdateAfterTraining);
end
