classdef RegressionAxesFactory < nnet.internal.cnn.ui.factory.AxesFactory
    % RegressionAxesFactory   Factory for creating the AxesViews and Metrics for regression version of the training plot
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    methods        
        function [axesView, metrics] = createMainAxesAndMetrics(~, epochInfo, epochDisplayer)      
            
            % RMSE metrics
            trainingUpdateableSeries = nnet.internal.cnn.ui.axes.UpdateableSeries();
            validationUpdateableSeries = nnet.internal.cnn.ui.axes.UpdateableSeries();
            
            % We need the validation metric to be able to update itself
            % after training has completed (e.g. for the final validation
            % accuracy, computed outside the training loop). The RMSE
            % metric should not update outside training, otherwise the
            % smoothed RMSE is calculated incorrectly.
            rmseMetricUpdatesAfterTraining = false;
            validationMetricUpdatesAfterTraining = true;
            metrics{1} = nnet.internal.cnn.ui.metric.Metric(trainingUpdateableSeries, 'RMSE', rmseMetricUpdatesAfterTraining);
            metrics{2} = nnet.internal.cnn.ui.metric.Metric(validationUpdateableSeries, 'ValidationRMSE', validationMetricUpdatesAfterTraining);
            
            % Series for each line
            trainingSeries = iNormalSeries(trainingUpdateableSeries);
            trainingTrendSeries = iMovingAverageSeries(trainingUpdateableSeries);
            validationSeries = iNormalSeries(validationUpdateableSeries);
            
            % Lines: training, training-trend, validation
            trainingLine = iTrainingRMSELineModel(trainingSeries);
            trainingTrendLine = iTrainingRMSETrendLineModel(trainingTrendSeries);
            validationLine = iValidationRMSELineModel(validationSeries);  
                 
            % axes
            allLineModels = {trainingLine, trainingTrendLine, validationLine};
            axesModel = iRMSEAxesModel(allLineModels, epochInfo);
            axesView = nnet.internal.cnn.ui.axes.MultilineAxesView(axesModel, epochDisplayer, 'REGRESSION_RMSE');
        end
        
        function [axesView, metrics] = createLossAxesAndMetrics(~, epochInfo, epochDisplayer)
            
            % Regression Loss metrics
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
            axesModel = iRegressionLossAxesModel(allLineModels, epochInfo);            
            axesView = nnet.internal.cnn.ui.axes.MultilineAxesView(axesModel, epochDisplayer, 'REGRESSION_LOSS');
        end
        
        % Legend
        function [cellOfLegendSectionNames, cellOfLegendSectionStructArrs] = createLegendInfo(~)
            cellOfLegendSectionNames = {};
            cellOfLegendSectionStructArrs = {};
            
            % RMSE
            cellOfLegendSectionNames{end+1} = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionLegendRMSESectionName');
            cellOfLegendSectionStructArrs{end+1} = [iTrainingRMSETrendLegendStruct(), iTrainingRMSELegendStruct(), iValidationRMSELegendStruct()];
            
            % Loss
            cellOfLegendSectionNames{end+1} = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionLegendLossSectionName');
            cellOfLegendSectionStructArrs{end+1} = [iTrainingLossTrendLegendStruct(), iTrainingLossLegendStruct(), iValidationLossLegendStruct()];
        end
    end
    
end

% RMSE helpers
function lineModel = iTrainingRMSELineModel(series)
color = nnet.internal.cnn.ui.factory.Colors.RegressionRMSELineColor;
lineColor = [color, iOpacity()];
markerFaceColor = nnet.internal.cnn.ui.factory.Colors.RegressionRMSEDotColor;
markerEdgeColor = markerFaceColor;
hasFinalPointAnnotation = false;
lineModel = iLineModel(...
    series, ...
    iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionTrainingRMSELineLegendName'), ...
    iTrainingLineStyle(), ...
    lineColor, ...
    iMarkerType(), ...
    markerFaceColor, ...
    markerEdgeColor, ...
    iMarkerOnlyAtEndStrategy(),...
    hasFinalPointAnnotation);
end

function s = iTrainingRMSELegendStruct()
color = nnet.internal.cnn.ui.factory.Colors.RegressionRMSELineColor;
markerFaceColor = nnet.internal.cnn.ui.factory.Colors.RegressionRMSEDotColor;
markerEdgeColor = markerFaceColor;
s = struct();
s.Text = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionLegendTrainingRMSESubsectionName');
s.LineStyle = iTrainingLineStyle();
s.LineWidth = iLineWidth();
s.LineColor = [color, iLegendOpacity()];
s.Marker = iMarkerType();
s.MarkerFaceColor = markerFaceColor;
s.MarkerEdgeColor = markerEdgeColor;
end

function lineModel = iTrainingRMSETrendLineModel(series)
color = nnet.internal.cnn.ui.factory.Colors.RegressionRMSELineColor;
lineColor = [color, iDefaultOpacity()];
markerFaceColor = 'none';
markerEdgeColor = 'none';
hasFinalPointAnnotation = false;
lineModel = iLineModel(...
    series, ...
    iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionTrainingRMSETrendLineLegendName'), ...
    iTrainingLineStyle(), ...
    lineColor, ...
    iTrendMarkerType(), ...
    markerFaceColor, ...
    markerEdgeColor, ...
    iNoMarkerStrategy(),...
    hasFinalPointAnnotation);
end

function s = iTrainingRMSETrendLegendStruct()
color = nnet.internal.cnn.ui.factory.Colors.RegressionRMSELineColor;
s = struct();
s.Text = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionLegendTrainingTrendRMSESubsectionName');
s.LineStyle = iTrainingLineStyle();
s.LineWidth = iLineWidth();
s.LineColor = [color, iDefaultOpacity()];
s.Marker = iTrendMarkerType();
s.MarkerFaceColor = 'none';
s.MarkerEdgeColor = 'none';
end

function lineModel = iValidationRMSELineModel(series)
color = nnet.internal.cnn.ui.factory.Colors.RegressionValidationRMSELineColor;
lineColor = [color, iDefaultOpacity()];
markerFaceColor = color;
markerEdgeColor = color;
hasFinalPointAnnotation = true;
lineModel = iLineModel(...
    series, ...
    iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionValidationRMSELineLegendName'), ...
    iValidationLineStyle(), ...
    lineColor, ...
    iMarkerType(), ...
    markerFaceColor, ...
    markerEdgeColor, ...
    iMarkerOnAllPointsStrategy(),...
    hasFinalPointAnnotation);
end

function s = iValidationRMSELegendStruct()
color = nnet.internal.cnn.ui.factory.Colors.RegressionValidationRMSELineColor;
s = struct();
s.Text = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionLegendValidationRMSESubsectionName');
s.LineStyle = iValidationLineStyle();
s.LineWidth = iLineWidth();
s.LineColor = [color, iDefaultOpacity()];
s.Marker = iMarkerType();
s.MarkerFaceColor = color;
s.MarkerEdgeColor = color;
end 

function axesModel = iRMSEAxesModel(cellOfLineModels, epochInfo)
xLabel = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionRMSEAxesXLabel');
yLabel = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionRMSEAxesYLabel');
minXLim = iComputeMinXLim(epochInfo);
minYLim = [0, 0];
axesModel = iAxesModel(cellOfLineModels, xLabel, yLabel, minXLim, minYLim, epochInfo);
end

% Regression loss helpers
function lineModel = iTrainingLossLineModel(series)
color = nnet.internal.cnn.ui.factory.Colors.RegressionLossLineColor;
lineColor = [color, iOpacity()];
markerFaceColor = nnet.internal.cnn.ui.factory.Colors.RegressionLossDotColor;
markerEdgeColor = markerFaceColor;
hasFinalPointAnnotation = false;
lineModel = iLineModel(...
    series, ...
    iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionTrainingLossLineLegendName'), ...
    iTrainingLineStyle(), ...
    lineColor, ...
    iMarkerType(), ...
    markerFaceColor, ...
    markerEdgeColor, ...
    iMarkerOnlyAtEndStrategy(),...
    hasFinalPointAnnotation);
end

function s = iTrainingLossLegendStruct()
color = nnet.internal.cnn.ui.factory.Colors.RegressionLossLineColor;
markerFaceColor = nnet.internal.cnn.ui.factory.Colors.RegressionLossDotColor;
markerEdgeColor = markerFaceColor;
s = struct();
s.Text = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionLegendTrainingLossSubsectionName');
s.LineStyle = iTrainingLineStyle();
s.LineWidth = iLineWidth();
s.LineColor = [color, iLegendOpacity()];
s.Marker = iMarkerType();
s.MarkerFaceColor = markerFaceColor;
s.MarkerEdgeColor = markerEdgeColor;
end

function lineModel = iTrainingLossTrendLineModel(series)
color = nnet.internal.cnn.ui.factory.Colors.RegressionLossLineColor;
lineColor = [color, iDefaultOpacity()];
markerFaceColor = 'none';
markerEdgeColor = 'none';
hasFinalPointAnnotation = false;
lineModel = iLineModel(...
    series, ...
    iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionTrainingLossTrendLineLegendName'), ...
    iTrainingLineStyle(), ...
    lineColor, ...
    iTrendMarkerType(), ...
    markerFaceColor, ...
    markerEdgeColor, ...
    iNoMarkerStrategy(),...
    hasFinalPointAnnotation);
end

function s = iTrainingLossTrendLegendStruct()
color = nnet.internal.cnn.ui.factory.Colors.RegressionLossLineColor;
s = struct();
s.Text = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionLegendTrainingTrendLossSubsectionName');
s.LineStyle = iTrainingLineStyle();
s.LineWidth = iLineWidth();
s.LineColor = [color, iDefaultOpacity()];
s.Marker = iTrendMarkerType();
s.MarkerFaceColor = 'none';
s.MarkerEdgeColor = 'none';
end

function lineModel = iValidationLossLineModel(series)
color = nnet.internal.cnn.ui.factory.Colors.RegressionValidationLossLineColor;
lineColor = [color, iDefaultOpacity()];
markerFaceColor = color;
markerEdgeColor = color;
hasFinalPointAnnotation = true;
lineModel = iLineModel(...
    series, ...
    iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionValidationLossLineLegendName'), ...
    iValidationLineStyle(), ...
    lineColor, ...
    iMarkerType(), ...
    markerFaceColor, ...
    markerEdgeColor, ...
    iMarkerOnAllPointsStrategy(),...
    hasFinalPointAnnotation);
end

function s = iValidationLossLegendStruct()
color = nnet.internal.cnn.ui.factory.Colors.RegressionValidationLossLineColor;
s = struct();
s.Text = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionLegendValidationLossSubsectionName');
s.LineStyle = iValidationLineStyle();
s.LineWidth = iLineWidth();
s.LineColor = [color, iDefaultOpacity()];
s.Marker = iMarkerType();
s.MarkerFaceColor = color;
s.MarkerEdgeColor = color;
end

function axesModel = iRegressionLossAxesModel(cellOfLineModels, epochInfo)
xLabel = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionLossAxesXLabel');
yLabel = iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:RegressionLossAxesYLabel');
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
window = 1;
series = nnet.internal.cnn.ui.axes.MovingAverageSeries(updateableSeries, window);
end

function series = iMovingAverageSeries(updateableSeries)
series = nnet.internal.cnn.ui.axes.LocalLinearRegressionSeries(updateableSeries, iWindow());
end

function lineModel = iLineModel(series, name, lineStyle, lineColor, markerType, markerFaceColor, markerEdgeColor, markerStrategy, hasFinalPointAnnotation)
lineModel = nnet.internal.cnn.ui.axes.SeriesLineModel(series, markerStrategy);
lineModel.Name = name;
lineModel.LineStyle = lineStyle;
lineModel.LineWidth = 1.3;
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
