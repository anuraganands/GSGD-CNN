classdef Legend < nnet.internal.cnn.ui.layout.LegendLayout
    % Legend   Shows a legend with checkboxes
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % AreCheckboxesVisible   (logical) Are all the checkboxes visible?
        AreCheckboxesVisible = true
    end
    
    properties(Dependent)
        % Parent   (graphics handle) The parent component
        Parent 
    end
    
    properties(SetAccess = private)
        % MainPanel   (matlab.ui.container.Panel)
        % The main panel representing the entire legend.
        MainPanel
        
        % Checkboxes  (cell of uicontrol) Checkboxes for each section
        Checkboxes
        
        % PreferredWidth   (integer) Preferred width of MainPanel in points
        PreferredWidth
        
        % PreferredHeight   (integer) Preferred height of MainPanel in points
        PreferredHeight
    end
    
    properties(Access = private)        
        % InnerPanel  (panel) The child panel of MainPanel where all the
        % elements are added. This allows the MainPanel to have a margin.
        InnerPanel
        
        % VerticalLayout   (nnet.internal.cnn.ui.layout.VerticalLayout)
        % Layout for laying all the individual elements in the legend.
        VerticalLayout
        
        % SubSectionPanels  (cell of panels)
        SubSectionPanels
    end
    
    methods
        function this = Legend()
            this.MainPanel = uipanel(...
                'Parent', [], ...
                'BorderType', 'line', ...
                'BackgroundColor', 'white', ...
                'Units', 'points', ...
                'HighlightColor', iBorderColor());
            
            this.PreferredWidth = iWidthInPoints();
            this.PreferredHeight = 100 * iGetPointsPerPixel();
            
            this.MainPanel.Position(3) = this.PreferredWidth;
            
            this.Checkboxes = {};
            this.SubSectionPanels = {};
            
            this.createInnerGUIComponents();
        end
        
        function parent = get.Parent(this)
            parent = this.MainPanel.Parent; 
        end
        
        function set.Parent(this, parent)
            this.MainPanel.Parent = parent; 
        end
        
        function set.AreCheckboxesVisible(this, tf)
            for i=1:numel(this.Checkboxes) %#ok<MCSUP>
                this.Checkboxes{i}.Visible = iBooleanToOnOrOff(tf);  %#ok<MCSUP>
            end
        end
        
        function addSection(this, sectionName, sectionStructArr)

            % add vertical spacer if there is already a section
            if numel(this.Checkboxes) > 0
                this.addTinyVerticalSpacer();
            end
            
            % add panel for the section header.
            this.addSectionHeader(sectionName);
            
            % add successive panels for each subsection
            for lineIndex=1:numel(sectionStructArr)
                this.addSubSection(sectionStructArr(lineIndex), lineIndex);
            end
            
            this.updateSizeOfLegend();
        end
    end
    
    methods(Access = private)
        function addSectionHeader(this, sectionName)
            headerPanel = uiflowcontainer('v0', 'Parent', [], 'FlowDirection', 'lefttoright', 'BackgroundColor', 'white', 'Margin', iVerySmallMargin());
            
            % add checkbox
            sectionIndex = numel(this.Checkboxes)+1;
            checkbox = iAddCheckbox(headerPanel, sectionIndex, this.AreCheckboxesVisible);
            this.Checkboxes{end+1} = checkbox;
            
            % add section title
            iAddSectionTitle(headerPanel, sectionIndex, sectionName);
            
            iAddHorizontalSpacer(headerPanel);
            
            % add the headerPanel to the VerticalLayout.
            this.VerticalLayout.add( headerPanel, iConstantWeight() );
        end
        
        function addSubSection(this, sectionStruct, lineIndex)
            % addSubSection   the subsection consists of a fixed space on the left, then an
            % axes object that fills the remaining horizontal space.
            subSectionPanel = uiflowcontainer('v0', 'Parent', [], 'FlowDirection', 'lefttoright', 'BackgroundColor', 'white', 'Margin', iVerySmallMargin());
            
            leftPaddingWidth = 25*iGetPointsPerPixel();
            leftPadding = uipanel('Parent', subSectionPanel, 'BorderType', 'none', 'BackgroundColor', 'white');
            leftPadding.WidthLimits = [leftPaddingWidth, leftPaddingWidth];
            
            axesPanel = uipanel('Parent', subSectionPanel, 'BorderType', 'none', 'BackgroundColor', 'white');
            this.addAxes(axesPanel, sectionStruct, this.PreferredWidth - leftPaddingWidth, lineIndex);
            
            % add subSectionPanel to various things.
            this.SubSectionPanels{end+1} = subSectionPanel;
            this.VerticalLayout.add( subSectionPanel, iConstantWeight() );
        end
        
        function addAxes(this, parent, sectionStruct, axesWidthInPixels, lineIndex)
            
            % the axes draws two things:
            %   1) a line with the correct properties
            %   2) a text label on the right of the axes
            ax = axes('Parent', parent, 'Units', 'points', 'XLim', [0,axesWidthInPixels], 'YLim', [0,1], 'Units', 'normalized', 'Position', [0,0,1,1]);
            axis(ax, 'off');
            
            % draw line in the axes.
            lineWidth = 60;
            sectionIndex = numel(this.Checkboxes);
            lineTag = ['NNET_CNN_TRAININGPLOT_LEGEND_SECTION', num2str(sectionIndex), '_LINE', num2str(lineIndex)];
            line(ax, 'XData', [0,0.5,1]*lineWidth, 'YData', [0.5,0.5,0.5], ...
                'LineStyle', sectionStruct.LineStyle, ...
                'LineWidth', sectionStruct.LineWidth, ...
                'Color', sectionStruct.LineColor, ...
                'Marker', sectionStruct.Marker, ...
                'MarkerIndices', 2, ...
                'MarkerFaceColor', sectionStruct.MarkerFaceColor, ...
                'MarkerEdgeColor', sectionStruct.MarkerEdgeColor, ...
                'Tag', lineTag);
            
            % add text label on the right of the axes.
            spaceBetweenAxesAndText = 6;
            textTag = ['NNET_CNN_TRAININGPLOT_LEGEND_SECTION', num2str(sectionIndex), '_TEXT', num2str(lineIndex)];
            text(lineWidth+spaceBetweenAxesAndText, 0.5, sectionStruct.Text, 'Parent', ax, 'FontSize', iFontSize(), 'Tag', textTag);
        end
        
        function addTinyVerticalSpacer(this)
            verticalSpacer = uipanel('Parent', [], 'BorderType', 'none', 'BackgroundColor', 'white');
            this.VerticalLayout.add( verticalSpacer, iVerySmallWeight() );
        end
        
        function updateSizeOfLegend(this)
            % compute the height that the section should have
            numElements = numel(this.Checkboxes) + numel(this.SubSectionPanels);
            numSpacers = numel(this.Checkboxes)-1;
            heightOfContents = iTextHeight()*iGetPointsPerPixel() * (numElements * iConstantWeight() + numSpacers * iVerySmallWeight());
            this.PreferredHeight = heightOfContents + 2*iMargin();
            % update the height of the legend
            this.MainPanel.Position(4) = this.PreferredHeight;
            this.InnerPanel.Position = [iMargin(), iMargin(), this.PreferredWidth - 2*iMargin(), heightOfContents]; 
            % update the width of the legend
            this.MainPanel.Position(3) = this.PreferredWidth;
        end
    end
    
    methods(Access = private)
        function createInnerGUIComponents(this)
            this.InnerPanel = uipanel(...
                'Parent', this.MainPanel, ...
                'BorderType', 'none', ...
                'BackgroundColor', 'white', ...
                'Units', 'points', ...
                'Position', [iMargin(), iMargin(), this.PreferredWidth - 2*iMargin(), this.PreferredHeight - 2*iMargin()]);
            
            this.createMainVerticalLayout(this.InnerPanel);
        end
        
        function createMainVerticalLayout(this, parent)
            this.VerticalLayout = nnet.internal.cnn.ui.layout.VerticalLayout(parent);
            this.VerticalLayout.MainPanel.BackgroundColor = 'white';
        end
    end
end

% helpers
function checkbox = iAddCheckbox(parent, sectionIndex, isVisible)
tag = sprintf('NNET_CNN_TRAININGPLOT_LEGEND_CHECKBOX%d', sectionIndex);
checkbox = uicontrol(...
    'Parent', parent, ...
    'Style', 'checkbox', ...
    'String', '', ...
    'BackgroundColor', 'white', ...
    'Tag', tag, ...
    'Visible', iBooleanToOnOrOff(isVisible));
iSetPreferredWidth(checkbox, checkbox.Position(4));  % ensure that checkbox has minimal width.
end

function sectionTitle = iAddSectionTitle(parent, sectionIndex, sectionName)
tag = sprintf('NNET_CNN_TRAININGPLOT_LEGEND_SECTION_TITLE%d', sectionIndex);

% we use axes and text to ensure that the text is center-aligned.
panel = uipanel('Parent', parent', 'BorderType', 'none', 'BackgroundColor', 'white');
ax = axes('Parent', panel, 'XLim', [0,1], 'YLim', [0,1], 'Units', 'normalized', 'Position', [0,0,1,1]);
axis(ax, 'off');
sectionTitle = text(0, 0.5, sectionName, 'Parent', ax, 'FontSize', iFontSize(), 'FontWeight', 'bold', 'Tag', tag);
end

function iAddHorizontalSpacer(parent)
uipanel('Parent', parent, 'BorderType', 'none', 'BackgroundColor', 'white');
end

function weight = iConstantWeight()
weight = 1.0;
end

function weight = iVerySmallWeight()
weight = 0.5;
end

function color = iBorderColor()
color = [0.8, 0.8, 0.8];
end

function pointsPerPixel = iGetPointsPerPixel()
pointsPerInch = 72;
pixelPerInch = get(groot, 'ScreenPixelsPerInch');
pointsPerPixel = pointsPerInch/pixelPerInch;
end

function margin = iMargin()
margin = 12 * iGetPointsPerPixel();
end

function margin = iVerySmallMargin()
margin = 0.00001;
end

function height = iTextHeight()
height = 23;
end

function fontSize = iFontSize()
fontSize = 9;
end

function str = iBooleanToOnOrOff(tf)
if tf
    str = 'on';
else
    str = 'off';
end
end

function iSetPreferredWidth(obj, width)
obj.WidthLimits = [width, width];
end

function width = iWidthInPoints()
width = 260;
end
