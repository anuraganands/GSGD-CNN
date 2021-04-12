classdef DeterminateProgressText < nnet.internal.cnn.ui.progress.DeterminateProgress
    % DeterminateProgressText   Shows progress as text
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % Parent   (graphics handle) The parent of MainPanel which holds
        % the progress bar.
        Parent

        % Value   (double) The current value. The minimum is 0.
        Value = 0
        
        % Maximum   (positive double) The maximum value.
        Maximum = 1
    end
    
    properties(SetAccess = private)
        % MainPanel   (uipanel) The main panel holding the progress bar
        MainPanel
        
        % PreferredHeight   (integer) The preferred height of MainPanel
        PreferredHeight = 30
    end
    
    properties(Access = private)
        % ProgressText   (uicontrol) The text control that indicates the
        % current progress
        ProgressText
        
        % StopButton   (com.mathworks.mwswing.MJButton) The button to stop
        % training
        StopButton
        
        % StopButtonListener   (listener) Listener on the StopButton
        StopButtonListener
    end
    
    methods
        function this = DeterminateProgressText()
            this.Value = 0;
            this.Maximum = 1;
            this.Parent = [];
            this.createGUIComponents();
        end
        
        function set.Value(this, value)
            this.Value = value;
            this.updateText();
        end
        
        function set.Maximum(this, maxValue)
            this.Maximum = maxValue;
            this.updateText();
        end
        
        function set.Parent(this, parent)
            this.Parent = parent;
            this.MainPanel.Parent = parent; %#ok<MCSUP>
        end
    end
    
    methods(Access = private)
        function createGUIComponents(this)
            this.createMainPanel(this.Parent);
            horizontalFlow = uiflowcontainer('v0', 'Parent', this.MainPanel, 'FlowDirection', 'lefttoright');
            
            this.createProgressText(horizontalFlow);
            
            iAddFixedWidthSpacer(horizontalFlow, 5);
            this.createStopButton(horizontalFlow);
            iAddFixedWidthSpacer(horizontalFlow, 5);
            
            this.updateText();
        end
        
        function createMainPanel(this, parent)
            this.MainPanel = uipanel(...
                'Parent', parent, ...
                'BorderType', 'none', ...
                'Tag', 'NNET_CNN_TRAININGPLOT_DETERMINATEPROGTEXT_MAINPANEL');
        end
        
        function createProgressText(this, parent)
            this.ProgressText = uicontrol(...
                'Parent', parent, ...
                'Style', 'text', ...
                'String', '', ...
                'FontSize', 9, ...
                'FontUnits', 'points', ...
                'HorizontalAlignment', 'left', ...
                'Tag', 'NNET_CNN_TRAININGPLOT_DETERMINATEPROGTEXT_PROGRESSTEXT');
            iSetPreferredHeight(this.ProgressText, this.PreferredHeight);
        end
        
        function createStopButton(this, parent)
            this.StopButton = iCreateStopButton(parent);
            this.addJavaActionListener(this.StopButton, @this.stopButtonClickedCallback);
        end
        
        function updateText(this)
            % The message displays something like "Training x".
            % However, currValue indicates the iteration has just
            % completed, not the iteration that we are going to train.
            % Therefore, we must +1 to currValue, unless we have already
            % reached the last iteration.
            currValueToShow = this.Value + 1;
            m = message('nnet_cnn:internal:cnn:ui:trainingplot:ProgressBarLabelWithoutMax', num2str(currValueToShow));
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
function iSetPreferredHeight(h, height)
h.HeightLimits = [height, height];
end

function iSetPreferredWidth(h, width)
h.WidthLimits = [width width];
end

function iAddFixedWidthSpacer(parent, width)
spacer = uipanel('Parent', parent', 'BorderType', 'none');
iSetPreferredWidth(spacer, width);
iSetPreferredHeight(spacer, 0);
end

% java helpers
function stopButton = iCreateStopButton(parent)
javaStopButton = javaObjectEDT('com.mathworks.mwswing.MJToggleButton', '');
[stopButton, stopButtonWrapper] = javacomponent(javaStopButton, [], parent);
stopButtonWrapper.Tag = 'NNET_CNN_TRAININGPLOT_DETERMINATEPROGTEXT_STOPBUTTON_JAVAWRAPPER';
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
