classdef DeterminateProgressBar < nnet.internal.cnn.ui.progress.DeterminateProgress
    % DeterminateProgressBar   A determinate progress bar implemented in HG
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % Parent   (graphics handle) The parent of MainPanel which holds
        % the progress bar.
        Parent = []
    end
    
    properties
        % Value   (double) The value between Minimum and Maximum
        % (inclusive) indicating where the progress bar is.
        Value = 0
        
        % Maximum   (double) The maximum value of the progress bar
        Maximum = 1
    end
    
    properties(SetAccess = private)        
        % MainPanel   (uipanel) The main panel holding the progress bar
        MainPanel
    end
    
    properties(Dependent, SetAccess = private)
        % PreferredHeight   (integer) The preferred height of MainPanel
        PreferredHeight
    end
    
    properties(Access = private)
        % Minimum   (double) The minimum value of the progress bar
        Minimum = 0
        
        % ProgressText   (uicontrol) The text control that indicates the
        % current progress
        ProgressText
        
        % ProgressTextHeight   (integer) Height of ProgressText
        ProgressTextHeight = 23
        
        % Axes   (axes) The axes on which to draw the progress bar
        Axes
        
        % OuterBorder   (patch) The patch object that draws the border
        % around the axes
        OuterBorder
        
        % Bar   (patch) The patch object that draws the moving bar
        Bar
        
        % BarHeight   (integer) Height of progress bar
        BarHeight = 22
        
        % StopButton   (com.mathworks.mwswing.MJButton) The button to stop
        % training
        StopButton
        
        % StopButtonListener   (listener) Listener on the StopButton
        StopButtonListener
    end
    
    methods
        function this = DeterminateProgressBar()
            this.createGUIComponents();
        end
        
        function set.Value(this, value)
            this.Value = value;
            this.update();
        end
        
        function set.Maximum(this, maxValue)
            this.Maximum = maxValue;
            this.issueWarningIfBoundsInvalid();
            this.update();
        end
        
        function set.Parent(this, parent)
            this.Parent = parent;
            this.MainPanel.Parent = parent; %#ok<MCSUP>
        end
        
        function h = get.PreferredHeight(this)
            h = this.ProgressTextHeight + this.BarHeight; 
        end
    end
    
    methods(Access = private)
        function issueWarningIfBoundsInvalid(this)
            if this.areBoundsInvalid()
                this.Minimum = 0;
                this.Maximum = 1;
                warning(message('nnet_cnn:internal:cnn:ui:trainingplot:GenericWarning'));
            end
        end
        
        function tf = areBoundsInvalid(this)
            tf = ~isnumeric(this.Maximum) || ~isreal(this.Maximum) || ...
                isempty(this.Maximum) || iIsNotScalar(this.Maximum) || ...
                isnan(this.Maximum) || isinf(this.Maximum) || ...
                this.Maximum <= 0;
        end
        
        function update(this)
            this.updateBar(); 
            this.updateText();
        end
        
        function createGUIComponents(this)
            this.createMainPanel(this.Parent);
            verticalFlow = uiflowcontainer('v0', 'Parent', this.MainPanel, 'FlowDirection', 'topdown', 'Margin', iTinyMargin());
            
            % First row
            this.createProgressText(verticalFlow);
            
            % Second row
            horizontalFlow = uiflowcontainer('v0', 'Parent', verticalFlow, 'FlowDirection', 'lefttoright', 'Margin', iTinyMargin());
            iSetPreferredHeight(horizontalFlow, this.BarHeight);
            % Prog bar
            this.createAxes(horizontalFlow);
            this.createOuterBorder();
            this.createBar();
            % Stop button
            iAddFixedWidthSpacer(horizontalFlow, 5);
            this.createStopButton(horizontalFlow);
            
            this.update();
        end
        
        function createMainPanel(this, parent)
            this.MainPanel = uipanel(...
                'Parent', parent, ...
                'BorderType', 'none', ...
                'Tag', 'NNET_CNN_TRAININGPLOT_DETERMINATEPROGBARANDTEXT_MAINPANEL');
        end
        
        function createProgressText(this, parent)
             this.ProgressText = uicontrol(...
                'Parent', parent, ...
                'Style', 'text', ...
                'String', '', ...
                'FontSize', 9, ...
                'FontUnits', 'points', ...
                'HorizontalAlignment', 'left', ...
                'Tag', 'NNET_CNN_TRAININGPLOT_DETERMINATEPROGBARANDTEXT_PROGRESSTEXT');
            iSetPreferredHeight(this.ProgressText, this.ProgressTextHeight);
        end
        
        function createAxes(this, parent)            
            % createAxes   Creates axes with no border and no tick marks.
            % Limits are set to [0,1] for x and y.
            
            panel = uipanel('Parent', parent, 'BorderType', 'none');
            this.Axes = axes(...
                'Parent', panel, ...
                'XGrid', 'off', 'YGrid', 'off', ...
                'XTick', [], 'YTick', [], ...
                'XLim', [0,1], 'YLim', [0,1], ...
                'Position', [0,0,1,1], ...
                'Box', 'off');
            this.Axes.XRuler.Visible = 'off';
            this.Axes.YRuler.Visible = 'off';
            
            iSetPreferredHeight(panel, this.BarHeight);
        end
        
        function createOuterBorder(this)
            this.OuterBorder = patch(this.Axes, ...
                'XData', [0,0,1,1], ...
                'YData', [0,1,1,0], ...
                'FaceColor', iLightGray(), ...
                'EdgeColor', iGreyColor(), ...
                'LineWidth', 0.1, ...
                'Tag', 'NNET_CNN_TRAININGPLOT_DETERMINATEPROGBARANDTEXT_OUTERBORDER');
        end
        
        function createBar(this)
            initialProgress = 0;
            this.Bar = patch(this.Axes, ...
                [0,0,initialProgress,initialProgress], ...
                [0,1,1,0], ...
                iMATLABBlue(), ...
                'EdgeColor', 'none', ...
                'Tag', 'NNET_CNN_TRAININGPLOT_DETERMINATEPROGBARANDTEXT_BAR');
        end
        
        function createStopButton(this, parent)
            
            % create vertical flow with spacers, to vertically center-align
            % the button
            verticalFlow = uiflowcontainer('v0', 'Parent', parent', 'FlowDirection', 'topdown', 'Margin', iTinyMargin());
            
            iAddFlexibleVerticalSpacer(verticalFlow);
            
            [this.StopButton, stopButtonWidth] = iCreateStopButton(verticalFlow);
            iSetPreferredWidth(verticalFlow, stopButtonWidth);
            this.addJavaActionListener(this.StopButton, @this.stopButtonClickedCallback);
            
            iAddFlexibleVerticalSpacer(verticalFlow);
        end
        
        function updateBar(this)
            value = iEnsureValueIsBetweenZeroAndMaximum(this.Value, this.Maximum);
            progressRatio = double(value) / double(this.Maximum);
            
            this.Bar.XData = [0,0,progressRatio,progressRatio];
        end
        
        function updateText(this)
            % The message displays something like "Training x of y".
            % However, currValue indicates the iteration has just
            % completed, not the iteration that we are going to train.
            % Therefore, we must +1 to currValue, unless we have already
            % reached the last iteration.
            currValue = iEnsureValueIsBetweenZeroAndMaximum(this.Value + 1, this.Maximum);
            m = message('nnet_cnn:internal:cnn:ui:trainingplot:ProgressBarLabel', num2str(currValue), num2str(this.Maximum));
            this.ProgressText.String = m.getString();
        end
        
        function addJavaActionListener(this, javaObject, callback)
            h = handle(javaObject, 'callbackproperties');
            this.StopButtonListener = handle.listener(h, 'ActionPerformed', callback);
        end
        
        % callbacks
        function stopButtonClickedCallback(this, ~, ~)
            this.notify('StopButtonClicked'); 
        end
    end
end

% helpers
function color = iGreyColor()
color = [188,188,188]/256;
end

function color = iMATLABBlue()
color = [0, 114, 189]/256;
end

function color = iLightGray()
color = [230,230,230]/256;
end

function value = iEnsureValueIsBetweenZeroAndMaximum(value, maximum)
value = min( max(value, 0), maximum );
end

function tf = iIsNotScalar(value)
tf = numel(value) ~= 1;
end

function iSetPreferredHeight(obj, height)
obj.HeightLimits = [height, height];
end

function iSetPreferredWidth(h, width)
h.WidthLimits = [width width];
end

function iAddFixedWidthSpacer(parent, width)
spacer = uipanel('Parent', parent', 'BorderType', 'none');
iSetPreferredWidth(spacer, width);
iSetPreferredHeight(spacer, 0);
end

function iAddFlexibleVerticalSpacer(parent)
spacer = uipanel('Parent', parent, 'BorderType', 'none');
iSetPreferredWidth(spacer, 0);
end

function margin = iTinyMargin()
margin = 0.0000001;
end

% java helpers
function [stopButton, stopButtonWidth] = iCreateStopButton(parent)
javaStopButton = javaObjectEDT('com.mathworks.mwswing.MJToggleButton', '');
[stopButton, stopButtonWrapper] = javacomponent(javaStopButton, [], parent);
stopButtonWrapper.Tag = 'NNET_CNN_TRAININGPLOT_DETERMINATEPROGBARANDTEXT_STOPBUTTON_JAVAWRAPPER';
stopButtonWidth = 18;
iSetPreferredWidth(stopButtonWrapper, stopButtonWidth);
iSetPreferredHeight(stopButtonWrapper, stopButtonWidth);

% remove border
stopButton.setBorder([]);
stopButton.setFocusPainted(false);
stopButton.setBorderPainted(false);
stopButton.setContentAreaFilled(false);
zeroSizedInsets = javaObjectEDT('java.awt.Insets',0,0,0,0);
stopButton.setMargin(zeroSizedInsets);
% make time between registered clicks a long time.
stopButton.setMultiClickThreshhold(iALongTime());

normalIcon = iIcon('stop_normal_16.png');
iSetNormalIcon(stopButton, normalIcon);
iSetRolloverIcon(stopButton, iIcon('stop_normal_hover_16.png'));
iSetSelectedIcon(stopButton, iIcon('stop_animated_waiting_16.gif'));
iSetSize(stopButton, normalIcon.getIconWidth(), normalIcon.getIconHeight());
end

function javaDim = iDimension(width, height)
javaDim = java.awt.Dimension(width, height);
end

function someTime = iALongTime()
someTime = uint32(2^32-1);
end

function iSetNormalIcon(javaObj, icon)
javaObj.setIcon(icon);
end

function iSetRolloverIcon(javaObj, icon)
javaObj.setRolloverIcon(icon);
end

function iSetSelectedIcon(javaObj, icon)
javaObj.setSelectedIcon(icon);
end

function iSetSize(javaObj, width, height)
sz = iDimension(width, height);
javaObj.setMinimumSize(sz);
javaObj.setMaximumSize(sz);
javaObj.setPreferredSize(sz);

% Workaround for MATLAB Online: there is a time lag in displaying the
% progress bar when setting it to be visible. To avoid this time lag, we
% set the size explicitly. See g1360174.
javaObj.setSize(sz);
end

function icon = iIcon(filename)
icon = toolpack.component.Icon(nnet.internal.resourcePath(filename));
icon = icon.Peer;
end

