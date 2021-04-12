classdef MultilineAxesView < nnet.internal.cnn.ui.axes.AxesView
    % MultilineAxesView   View of an multiline axes
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % Panel   (uipanel) The parent panel of the axes
        Panel
    end
    
    properties(Access = private)
        % AxesModel   (nnet.internal.cnn.ui.axes.AxesModel)
        AxesModel
        
        % Axes   (axes) The main axes
        Axes
        
        % EpochRectangles   (patch) The patch object representing the epoch
        % rectangle backgrounds.
        EpochRectangles
        
        % EpochTexts   (array of text) The array of text objects that label
        % the epochs
        EpochTexts
        
        % EpochDisplayer   (nnet.internal.cnn.ui.axes.EpochDisplayer)
        % Helper class for drawing and updating the epochs
        EpochDisplayer
        
        % Lines   (cell of line) The lines added to the Axes
        Lines
        
        % TagSuffix (string) The tag to be added to graphics objects
        % created by this AxesView.
        TagSuffix
    end
    
    properties(Constant, Access = private)
        % AnnotationMarkerSize (int)
        % Size of markers added as annotations, e.g. after training.
        AnnotationMarkerSize = 10;
        
        % AnnotationMarkerSize (int)
        % Style of markers added as annotations, e.g. after training.
        AnnotatationMarkerStyle = 'o';
        
        % Color of markers added as annotations, e.g. after training.
        AnnotationMarkerColor = [0, 0, 0];
    end
    
    methods
        function this = MultilineAxesView(axesModel, epochDisplayer, tagSuffix)
            this.Panel = uipanel('Parent', [], 'BorderType', 'none', 'Tag', 'NNET_CNN_TRAININGPLOT_AXESVIEW_PANEL');
            this.AxesModel = axesModel;
            this.EpochDisplayer = epochDisplayer;
            this.TagSuffix = tagSuffix;            
            
            this.createGUIComponents(tagSuffix);
        end
        
        function update(this)
            % compute the bounds once only.
            xLim = this.AxesModel.XLim;
            yLim = this.AxesModel.YLim;
            
            % update everything 
            this.updateBounds(xLim, yLim);
            this.updateEpochs(xLim, yLim);
            this.updateLines();
        end
        
        function finalize(this)
            % Called when the training has been completed.
            %
            % This will add annotations on the final point to any
            % LineModels that require it (i.e. LineModels for validation
            % data).
            
            for i=1:numel(this.AxesModel.LineModels)
                lineModel = this.AxesModel.LineModels{i};
                
                % Draw annotation only if the LineModel requires it and
                % contains data to draw.
                if lineModel.HasFinalPointAnnotation && ~isempty(lineModel.XValues)
                    % Update the actual line with the final point.
                    this.Lines{i}.XData = lineModel.XValues;
                    this.Lines{i}.YData = lineModel.YValues;
                    this.Lines{i}.MarkerIndices = lineModel.MarkerIndices;
                    
                    % Add extra point and text.
                    this.addLabelledPoint(double(lineModel.XValues(end)),...
                        double(lineModel.YValues(end)),...
                        iMessageString('nnet_cnn:internal:cnn:ui:trainingplot:FinalValidationPointLabel'));
                end
            end
            
        end
    end
    
    methods(Access = private)
        function createGUIComponents(this, tagSuffix)
            tag = sprintf('NNET_CNN_TRAININGPLOT_AXESVIEW_AXES_%s', tagSuffix);
            
            this.Axes = axes(...
                'Parent', this.Panel, ...
                'Tag', tag, ...
                'XGrid', 'off', ...
                'YGrid', 'on', ...
                'Position', [0.07 0.15 0.86 0.78]);
            
            this.setLabels();
            
            xLim = this.AxesModel.XLim;
            yLim = this.AxesModel.YLim;
            
            this.updateBounds(xLim, yLim);
            
            this.ensureChildObjectsCanBeClipped();
            this.createEpochs();
            this.updateEpochs(xLim, yLim);
            
            this.createLines();
            this.updateLines();
            
            drawnow();
        end
        
        % lines
        function createLines(this)
            this.Lines = {};
            for i=1:numel(this.AxesModel.LineModels)
                lineModel = this.AxesModel.LineModels{i};
                tagSuffix = num2str(i);
                this.Lines{i} = iCreateLine(this.Axes, lineModel, tagSuffix);
            end
        end
        
        function updateLines(this)
            for i=1:numel(this.AxesModel.LineModels)
                lineModel = this.AxesModel.LineModels{i};
                this.Lines{i}.XData = lineModel.XValues;
                this.Lines{i}.YData = lineModel.YValues;
                this.Lines{i}.MarkerIndices = lineModel.MarkerIndices;
            end
        end
        
        % bounds
        function updateBounds(this, xLim, yLim)
            xlim(this.Axes, xLim);
            ylim(this.Axes, yLim);
        end
        
        % labels
        function setLabels(this)
            xlabel(this.Axes, this.AxesModel.XLabel, 'Interpreter', 'none');
            ylabel(this.Axes, this.AxesModel.YLabel, 'Interpreter', 'none');
        end
        
        % epochs
        function createEpochs(this)
            this.EpochRectangles = patch(this.Axes, 'XData', [], 'YData', [], 'Tag', 'NNET_CNN_TRAININGPLOT_AXESVIEW_EPOCHRECTANGLES');
            this.EpochTexts = matlab.graphics.primitive.Text.empty(0,1);
            for i=1:numel(this.AxesModel.EpochIndicesForTexts)
                this.EpochTexts(end+1) = text(this.Axes, 'String', '', 'Tag', 'NNET_CNN_TRAININGPLOT_AXESVIEW_EPOCHTEXTS'); 
            end
            this.EpochDisplayer.initializeEpochTexts(this.EpochTexts);
        end
        
        function updateEpochs(this, xLim, yLim)
            this.EpochDisplayer.updateEpochRectangles(...
                this.EpochRectangles, ...
                this.AxesModel.EpochInfo.NumEpochs, ...
                this.AxesModel.EpochInfo.NumItersPerEpoch, ...
                yLim);
            
            this.EpochDisplayer.updateEpochTexts(...
                this.EpochTexts, ...
                this.AxesModel.EpochIndicesForTexts, ...
                this.AxesModel.EpochInfo.NumItersPerEpoch, ...
                xLim, ...
                yLim);
        end
        
        % clipping
        function ensureChildObjectsCanBeClipped(this)
            % ensureChildObjectsCanBeClipped   Turning on 'Clipping'
            % ensures that each child object can control whether it gets
            % clipped by the Axes or not. Setting ClippingStyle to
            % 'rectangle' allows the part of the child that is within the
            % bounds of the axes to be shown, with the rest clipped off.
            this.Axes.Clipping = 'on';
            this.Axes.ClippingStyle = 'rectangle';
        end
        
        % Annotation for final point
        function addLabelledPoint(this, x, y, labelText)
            
            % Note that the final validation values may be singles,
            % which aren't supported by text(), so cast them.
            x = double(x);
            y = double(y);
            
            line(x, y,...
                'MarkerSize', this.AnnotationMarkerSize,...
                'Marker', this.AnnotatationMarkerStyle,...
                'MarkerEdgeColor', this.AnnotationMarkerColor,...
                'Tag', "NNET_CNN_TRAININGPLOT_AXESVIEW_ANNOTATIONMARKER_" + this.TagSuffix,...
                'Parent', this.Axes);

            t = text(x, y, iLeftPadWithWhiteSpace(labelText),...
                'Tag', "NNET_CNN_TRAININGPLOT_AXESVIEW_ANNOTATIONLABEL_" + this.TagSuffix,...
                'Parent', this.Axes,...
                'HorizontalAlignment', 'left');
            
            iFitTextInXAxis(this.Axes, t);
        end
    end
end

% helpers
function l = iCreateLine(parent, lineModel, tagSuffix)
tag = ['NNET_CNN_TRAININGPLOT_AXESVIEW_LINE_', tagSuffix];
l = line('Parent', parent, 'Tag', tag);
l.LineStyle  = lineModel.LineStyle;
l.LineWidth  = lineModel.LineWidth;
l.Color      = lineModel.LineColor;
l.Marker     = lineModel.MarkerType;
l.MarkerSize = lineModel.MarkerSize;
l.MarkerFaceColor = lineModel.MarkerFaceColor;
l.MarkerEdgeColor = lineModel.MarkerEdgeColor;

l.XData = lineModel.XValues;
l.YData = lineModel.YValues;
l.MarkerIndices = lineModel.MarkerIndices;
end

function str = iMessageString(varargin)
m = message(varargin{:});
str = m.getString();
end

function str = iLeftPadWithWhiteSpace(str)
str = "   " + str;
end

function iFitTextInXAxis(ax, t)
% Guarantees that a text object t will fit in an axes ax, by shifting the
% upper xlim to accommodate it.

% Calculate how far t extends beyonds the xlims.
rightOverhang = (t.Extent(1) + t.Extent(3)) - ax.XLim(2);

% Only rescale if we need to.
if rightOverhang > 0
    % Account for the fact that we're measuring the text extent in data
    % units, but we care about the pixel size of the text.
    oldXWidth = diff(ax.XLim);
    newXWidth = (ax.XLim(2) + rightOverhang) - ax.XLim(1);
    
    % Applying this rescale factor accounts for the difference in
    % pixel/data unit ratio before and after rescaling.
    dataToPxRescale = newXWidth/oldXWidth;
    
    % Apply a safety factor to make sure we're adding enough to the upper xlim.
    %With safetyFactor = 1.2, we add 20 % more than we need.
    safetyFactor = 1.2;
    
    ax.XLim = [ax.XLim(1), ax.XLim(2) + safetyFactor * dataToPxRescale * rightOverhang];
end

end
