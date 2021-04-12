classdef ClassificationAxesFactory < nnet.internal.cnn.ui.factory.AxesFactory
    % ClassificationAxesFactory   Factory for creating the AxesViews and Metrics for Classification version of the training plot
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    methods
        function [axesView, metrics] = createMainAxesAndMetrics(~, epochInfo, epochDisplayer) 

            % Accuracy metrics
            trainingUpdateableSeries = nnet.internal.cnn.ui.axes.UpdateableSeries();
            validationUpdateableSeries = nnet.internal.cnn.ui.axes.UpdateableSeries();
            
            % We need the validation metric to be able to update itself
            % after training has completed (e.g. for the final validation
            % accuracy, computed outside the training loop). The accuracy
            % metric should not update outside training, otherwise the
            % smoothed accuracy is calculated incorrectly.
            accuracyMetricUpdatesAfterTraining = false;
            validationMetricUpdatesAfterTraining = true;
            metrics{1} = nnet.internal.cnn.ui.metric.Metric(trainingUpdateableSeries, 'Accuracy', accuracyMetricUpdatesAfterTraining);
            metrics{2} = nnet.internal.cnn.ui.metric.Metric(validationUpdateableSeries, 'ValidationAccuracy', validationMetricUpdatesAfterTraining);
            
            % Series for each line
            trainingSeries = iNormalSeries(trainingUpdateableSeries);
            trainingTrendSeries = iMovingAverageSeries(trainingUpdateableSeries);
            validationSeries = iNormalSeries(validationUpdateableSeries);
            
            % Lines: training, training-trend, validation
            trainingLine = iTrainingAccuracyLineModel(trainingSeries);
            trainingTrendLine = iTrainingAccuracyTrendLineModel(trainingTrendSeries);
            validationLine = iValidationAccuracyLineModel(validationSeries);
                 
            % axes
            allLineModels = {trainingLine, trainingTrendLine, validationLine};
            axesModel = iAccuracyAxesModel(allLineModels, epochInfo);
            axesView = nnet.internal.cnn.ui.axes.MultilineAxesView(axesModel, epochDisplayer, 'CLASSIFICATION_ACCURACY');
        end
        
        function [axesView, metrics] = createLossAxesAndMetrics(~, epochInfo, epochDisplayer)
            
            % Classification Loss metrics
            trainingUpdateableSeries = nnet.internal.cnn.ui.axes.UpdateableSeries();
            validationUpdateableSeries = nnet.internal.cnn.ui.axes.UpdateableSeries(); 

            % As for this.createMainAxesAndMetrics, the validation metric
            % must be able to update after training, but the loss metric
            % should not be able to.
            lossMetricUpdatesAfterTraining = false;
            validationMetricUpdatesAfterTraining = true;
            metrics{1} = nnet.internal.cnn.ui.metric.Metric(trainingUpdateableSeries, 'Loss', lossMetricUpdatesAfterTraining);
            metrics{2} = nnet.internal.cnn.ui.metric.Metric(validationUpdateableSeries, 'ValidationLoss', validationMetricUpdatesAfterTraining);
            
            % Series for each line
            trainingSeries = iNormalSeries(trainingUpdateableSeries);
            trainingTrendSeries = iMovingAverageSeries(trainingUpdateableSeries);
            validationSeries = iNormalSeries(validationUpdateableSeries);
            
            % Lines: training, training-trend, validation
            trainingLine = iTrainingLossLineModel(trainingSeries);
            trainingTrendLine = iTrainingLossTrendLineModel(trainingTrendSeries);
            validationLine = iValidationLossLineModel(validationSeries);
                 
            % axes
            allLineModels = {trainingLine, trainingTrendLine, validationLine};
            axesModel = iClassificationLossAxesModel(allLineModels, epochInfo);            
            axesView = nnet.internal.cnn.ui.axes.MultilineAxesView(axesModel, epochDisplayer, 'CLASSIFICATION_LOSS');
        end
        
        % Legend
        function [cellOfLegendSectionNames, cellOfLegendSectionStructArrs] = createLegendInfo(~)
            cellOfLegendSectionNames = {};
            cellOfLegendSectionStructArrs = {};
            
            % Accuracy
            cellOfLegendSectionNames{end+1} = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationLegendAccuracySectionName');
            cellOfLegendSectionStructArrs{end+1} = [iTrainingAccuracyTrendLegendStruct(), iTrainingAccuracyLegendStruct(), iValidationAccuracyLegendStruct()];
            
            % Loss
            cellOfLegendSectionNames{end+1} = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationLegendLossSectionName');
            cellOfLegendSectionStructArrs{end+1} = [iTrainingLossTrendLegendStruct(), iTrainingLossLegendStruct(), iValidationLossLegendStruct()];
        end
    end
    
end

% Accuracy helpers
function lineModel = iTrainingAccuracyLineModel(series)
color = nnet.internal.cnn.ui.factory.Colors.ClassificationAccuracyLineColor;
lineColor = [color, iOpacity()];
markerFaceColor = nnet.internal.cnn.ui.factory.Colors.ClassificationAccuracyDotColor;
markerEdgeColor = markerFaceColor;
hasFinalPointAnnotation = false;
lineModel = iLineModel(...
    series, ...
    iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationTrainingAccuracyLineLegendName'), ...
    iTrainingLineStyle(), ...
    lineColor, ...
    iMarkerType(), ...
    markerFaceColor, ...
    markerEdgeColor, ...
    iMarkerOnlyAtEndStrategy(),...
    hasFinalPointAnnotation);
end

function s = iTrainingAccuracyLegendStruct()
color = nnet.internal.cnn.ui.factory.Colors.ClassificationAccuracyLineColor;
markerFaceColor = nnet.internal.cnn.ui.factory.Colors.ClassificationAccuracyDotColor;
markerEdgeColor = markerFaceColor;
s = struct();
s.Text = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationLegendTrainingAccuracySubsectionName');
s.LineStyle = iTrainingLineStyle();
s.LineWidth = iLineWidth();
s.LineColor = [color, iLegendOpacity()];
s.Marker = iMarkerType();
s.MarkerFaceColor = markerFaceColor;
s.MarkerEdgeColor = markerEdgeColor;
end

function lineModel = iTrainingAccuracyTrendLineModel(series)
color = nnet.internal.cnn.ui.factory.Colors.ClassificationAccuracyLineColor;
lineColor = [color, iDefaultOpacity()];
markerFaceColor = 'none';
markerEdgeColor = 'none';
hasFinalPointAnnotation = false;
lineModel = iLineModel(...
    series, ...
    iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationTrainingAccuracyTrendLineLegendName'), ...
    iTrainingLineStyle(), ...
    lineColor, ...
    iTrendMarkerType(), ...
    markerFaceColor, ...
    markerEdgeColor, ...
    iNoMarkerStrategy(),...
    hasFinalPointAnnotation);
end

function s = iTrainingAccuracyTrendLegendStruct()
color = nnet.internal.cnn.ui.factory.Colors.ClassificationAccuracyLineColor;
s = struct();
s.Text = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationLegendTrainingTrendAccuracySubsectionName');
s.LineStyle = iTrainingLineStyle();
s.LineWidth = iLineWidth();
s.LineColor = [color, iDefaultOpacity()];
s.Marker = iTrendMarkerType();
s.MarkerFaceColor = 'none';
s.MarkerEdgeColor = 'none';
end

function lineModel = iValidationAccuracyLineModel(series)
color = nnet.internal.cnn.ui.factory.Colors.ClassificationValidationAccuracyLineColor;
lineColor = [color, iDefaultOpacity()];
markerFaceColor = color;
markerEdgeColor = color;
hasFinalPointAnnotation = true;
lineModel = iLineModel(...
    series, ...
    iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationValidationAccuracyLineLegendName'), ...
    iValidationLineStyle(), ...
    lineColor, ...
    iMarkerType(), ...
    markerFaceColor, ...
    markerEdgeColor, ...
    iMarkerOnAllPointsStrategy(),...
    hasFinalPointAnnotation);
end

function s = iValidationAccuracyLegendStruct()
color = nnet.internal.cnn.ui.factory.Colors.ClassificationValidationAccuracyLineColor;
s = struct();
s.Text = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationLegendValidationAccuracySubsectionName');
s.LineStyle = iValidationLineStyle();
s.LineWidth = iLineWidth();
s.LineColor = [color, iDefaultOpacity()];
s.Marker = iMarkerType();
s.MarkerFaceColor = color;
s.MarkerEdgeColor = color;
end 

function axesModel = iAccuracyAxesModel(cellOfLineModels, epochInfo)
xLabel = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationAccuracyAxesXLabel');
yLabel = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationAccuracyAxesYLabel');
minXLim = iComputeMinXLim(epochInfo);
minYLim = [0, 105];
axesModel = iAxesModel(cellOfLineModels, xLabel, yLabel, minXLim, minYLim, epochInfo);
end

% Classification Loss helpers
function lineModel = iTrainingLossLineModel(series)
color = nnet.internal.cnn.ui.factory.Colors.ClassificationLossLineColor;
lineColor = [color, iOpacity()];
markerFaceColor = nnet.internal.cnn.ui.factory.Colors.ClassificationLossDotColor;
markerEdgeColor = markerFaceColor;
hasFinalPointAnnotation = false;
lineModel = iLineModel(...
    series, ...
    iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationTrainingLossLineLegendName'), ...
    iTrainingLineStyle(), ...
    lineColor, ...
    iMarkerType(), ...
    markerFaceColor, ...
    markerEdgeColor, ...
    iMarkerOnlyAtEndStrategy(),...
    hasFinalPointAnnotation);
end

function s = iTrainingLossLegendStruct()
color = nnet.internal.cnn.ui.factory.Colors.ClassificationLossLineColor;
markerFaceColor = nnet.internal.cnn.ui.factory.Colors.ClassificationLossDotColor;
markerEdgeColor = markerFaceColor;
s = struct();
s.Text = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationLegendTrainingLossSubsectionName');
s.LineStyle = iTrainingLineStyle();
s.LineWidth = iLineWidth();
s.LineColor = [color, iLegendOpacity()];
s.Marker = iMarkerType();
s.MarkerFaceColor = markerFaceColor;
s.MarkerEdgeColor = markerEdgeColor;
end

function lineModel = iTrainingLossTrendLineModel(series)
color = nnet.internal.cnn.ui.factory.Colors.ClassificationLossLineColor;
lineColor = [color, iDefaultOpacity()];
markerFaceColor = 'none';
markerEdgeColor = 'none';
hasFinalPointAnnotation = false;
lineModel = iLineModel(...
    series, ...
    iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationTrainingLossTrendLineLegendName'), ...
    iTrainingLineStyle(), ...
    lineColor, ...
    iTrendMarkerType(), ...
    markerFaceColor, ...
    markerEdgeColor, ...
    iNoMarkerStrategy(),...
    hasFinalPointAnnotation);
end

function s = iTrainingLossTrendLegendStruct()
color = nnet.internal.cnn.ui.factory.Colors.ClassificationLossLineColor;
s = struct();
s.Text = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationLegendTrainingTrendLossSubsectionName');
s.LineStyle = iTrainingLineStyle();
s.LineWidth = iLineWidth();
s.LineColor = [color, iDefaultOpacity()];
s.Marker = iTrendMarkerType();
s.MarkerFaceColor = 'none';
s.MarkerEdgeColor = 'none';
end

function lineModel = iValidationLossLineModel(series)
color = nnet.internal.cnn.ui.factory.Colors.ClassificationValidationLossLineColor;
lineColor = [color, iDefaultOpacity()];
markerFaceColor = color;
markerEdgeColor = color;
hasFinalPointAnnotation = true;
lineModel = iLineModel(...
    series, ...
    iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationValidationLossLineLegendName'), ...
    iValidationLineStyle(), ...
    lineColor, ...
    iMarkerType(), ...
    markerFaceColor, ...
    markerEdgeColor, ...
    iMarkerOnAllPointsStrategy(),...
    hasFinalPointAnnotation);
end

function s = iValidationLossLegendStruct()
color = nnet.internal.cnn.ui.factory.Colors.ClassificationValidationLossLineColor;
s = struct();
s.Text = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationLegendValidationLossSubsectionName');
s.LineStyle = iValidationLineStyle();
s.LineWidth = iLineWidth();
s.LineColor = [color, iDefaultOpacity()];
s.Marker = iMarkerType();
s.MarkerFaceColor = color;
s.MarkerEdgeColor = color;
end

function axesModel = iClassificationLossAxesModel(cellOfLineModels, epochInfo)
xLabel = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationLossAxesXLabel');
yLabel = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:ClassificationLossAxesYLabel');
minXLim = iComputeMinXLim(epochInfo);
minYLim = [0, 0];
axesModel = iAxesModel(cellOfLineModels, xLabel, yLabel, minXLim, minYLim, epochInfo);
end

% low-level helpers
function messageString = iMessageString(varargin)
m = message(varargin{:});
messageString = m.getString();
end

function series = iNormalSeries(updateableSeries)
series = nnet.internal.cnn.ui.axes.MovingAverageSeries(updateableSeries, 1);
end

function series = iMovingAverageSeries(updateableSeries)
series = nnet.internal.cnn.ui.axes.LocalLinearRegressionSeries(updateableSeries, iWindow());
end

function lineModel = iLineModel(series, name, lineStyle, lineColor, markerType, markerFaceColor, markerEdgeColor, markerStrategy, hasFinalPointAnnotation)
lineModel = nnet.internal.cnn.ui.axes.SeriesLineModel(series, markerStrategy);
lineModel.Name = name;
lineModel.LineStyle = lineStyle;
lineModel.LineWidth = iLineWidth();
lineModel.LineColor = lineColor;
lineModel.MarkerType = markerType;
lineModel.MarkerSize = 5;
lineModel.MarkerFaceColor = markerFaceColor;
lineModel.MarkerEdgeColor = markerEdgeColor;
lineModel.HasFinalPointAnnotation = hasFinalPointAnnotation;
end

function axesModel = iAxesModel(cellOfLineModels, xLabel, yLabel, minXLim, minYLim, epochInfo)
axesModel = nnet.internal.cnn.ui.axes.MultilineAxesModel(cellOfLineModels, xLabel, yLabel, minXLim, minYLim, epochInfo);
end

function markerStrategy = iMarkerOnlyAtEndStrategy()
markerStrategy = nnet.internal.cnn.ui.axes.MarkerOnlyAtEndStrategy();
end

function markerStrategy = iNoMarkerStrategy()
markerStrategy = nnet.internal.cnn.ui.axes.NoMarkerStrategy();
end

function markerStrategy = iMarkerOnAllPointsStrategy()
markerStrategy = nnet.internal.cnn.ui.axes.MarkerOnAllPointsStrategy();
end

function window = iWindow()
window = 15;
end

function lim = iComputeMinXLim(epochInfo)
numIters = epochInfo.NumIters;
lim = [0, min(100, numIters)];
end

function width = iLineWidth()
width = 1.3;
end

function lineStyle = iTrainingLineStyle()
lineStyle = '-';
end

function lineStyle = iValidationLineStyle()
lineStyle = '--';
end

function alpha = iOpacity()
alpha = 0.2;
end

function alpha = iLegendOpacity()
alpha = 0.4;
end

function alpha = iDefaultOpacity()
alpha = 1.0;
end

function markerType = iMarkerType()
markerType = 'o';
end

function markerType = iTrendMarkerType()
markerType = 'none';
end

