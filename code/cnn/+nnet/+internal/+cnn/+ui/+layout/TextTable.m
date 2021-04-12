classdef TextTable < nnet.internal.cnn.ui.layout.TextLayout
    % TextTable   Panel for showing text in tabular form
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Parent   (graphics handle) The parent component
        Parent 
    end
    
    properties(SetAccess = private)        
        % MainPanel   (matlab.ui.container.Panel)
        % The main panel representing the entire legend.
        MainPanel
        
        % PreferredHeight   (integer) The preferred height of the MainPanel
        % in points
        PreferredHeight
    end
    
    properties(Access = private)
        % GridContainer   (uigridcontainer)
        % The grid container that lays out all the text
        GridContainer
        
        % VerticalWeights   (vector of doubles) The relative heights of
        % each row
        VerticalWeights
        
        % CurrentSectionIndex   (integer) Used to generate tags.
        CurrentSectionIndex
        
        % SectionNames   (cellstring) Cell of section names.
        SectionNames
        
        % SectionStructs   (cell of struct arrays) This stores the section
        % structs
        SectionStructs
        
        % TextControls   (cell of nx2 array of uicontrols) Stores all the
        % uicontrols. Each cell element corresponds to a section and is of
        % size nx2 where n is the number of rows in that section. n varies
        % with the section in general.
        TextControls
    end
    
    methods
        function this = TextTable()
            this.MainPanel = uipanel(...
                'Parent', [], ...
                'BorderType', 'none');
            
            this.reset();
        end
        
        function parent = get.Parent(this)
            parent = this.MainPanel.Parent; 
        end
        
        function set.Parent(this, parent)
            this.MainPanel.Parent = parent;
        end
        
        function reset(this)
            this.CurrentSectionIndex = 0;
            this.SectionNames = {};
            this.SectionStructs = {};
            this.TextControls = {};
            
            this.VerticalWeights = [];
            this.PreferredHeight = 0;
            
            delete(this.GridContainer);
            this.GridContainer = uigridcontainer('v0', 'Parent', this.MainPanel, 'GridSize', [1,2]); % Cannot set any elements of GridSize to 0.
            this.GridContainer.Margin = iTinyMargin();
        end
        
        function addSection(this, sectionName, sectionStructArr)
            numContentRows = numel(sectionStructArr);
            this.CurrentSectionIndex = this.CurrentSectionIndex + 1;
            this.SectionNames{end+1} = sectionName;
            this.SectionStructs{end+1} = sectionStructArr;
            this.TextControls{end+1} = matlab.ui.control.UIControl.empty(0,2);
            
            % update size of grid container
            numRowsForTitle = 1;
            if this.CurrentSectionIndex == 1
                numRowsForSeparation = 0;
                this.GridContainer.GridSize(1) = numRowsForSeparation + numRowsForTitle + numContentRows;
            else
                numRowsForSeparation = 1;
                numNewRows = numRowsForSeparation + numRowsForTitle + numContentRows;
                this.GridContainer.GridSize(1) = this.GridContainer.GridSize(1) + numNewRows;
            end
            
            % add row for separation
            if numRowsForSeparation > 0
                this.addSeparationRow();
            end
            
            % add title and contents
            this.addTitle(sectionName);
            for i=1:numContentRows
                 this.addRow(i, sectionStructArr);
            end
            
            % update preferred height
            this.PreferredHeight = this.PreferredHeight + iGetPointsPerPixel() * iTextHeight() * (numRowsForTitle+numContentRows + 0.5 * numRowsForSeparation);
            % update GridContainer's vertical weights
            this.GridContainer.VerticalWeight = this.VerticalWeights;
        end
        
        function update(this, rowID, newRightText)
            [sectionIndex, subSectionIndex] = this.findRow(rowID);
            if ~isempty(sectionIndex) && ~isempty(subSectionIndex)
                this.SectionStructs{sectionIndex}(subSectionIndex).RightText = newRightText;
                this.TextControls{sectionIndex}(subSectionIndex, 2).String = newRightText;
            end
        end
    end
    
    methods(Access = private)
        function addSeparationRow(this)
            iAddSpacer(this.GridContainer);
            iAddSpacer(this.GridContainer);
            this.VerticalWeights = [this.VerticalWeights, 0.5];
        end
        
        function addTitle(this, titleName)
            uicontrol(...
                'Parent', this.GridContainer, ...
                'Style', 'text', ...
                'String', titleName, ...
                'FontWeight', 'bold', ...
                'FontSize', 9, ...
                'FontUnits', 'points', ...
                'HorizontalAlignment', 'left', ...
                'Tag', ['NNET_CNN_TRAININGPLOT_TEXTTABLE_TITLE', num2str(this.CurrentSectionIndex)]);
            
            iAddSpacer(this.GridContainer);
            this.VerticalWeights = [this.VerticalWeights, 1];
        end
        
        function addRow(this, rowIndex, sectionStructArr)
            leftControl = uicontrol(...
                'Parent', this.GridContainer, ...
                'Style', 'text', ...
                'String', sectionStructArr(rowIndex).LeftText, ...
                'FontSize', 9, ...
                'FontUnits', 'points', ...
                'HorizontalAlignment', 'left', ...
                'Tag', ['NNET_CNN_TRAININGPLOT_TEXTTABLE_LEFTTEXT_', sectionStructArr(rowIndex).RowID]);
            
            rightControl = uicontrol(...
                'Parent', this.GridContainer, ...
                'Style', 'text', ...
                'String', sectionStructArr(rowIndex).RightText, ...
                'FontSize', 9, ...
                'FontUnits', 'points', ...
                'HorizontalAlignment', 'left', ...
                'Tag', ['NNET_CNN_TRAININGPLOT_TEXTTABLE_RIGHTTEXT_', sectionStructArr(rowIndex).RowID]);
            
            this.TextControls{this.CurrentSectionIndex}(end+1, :) = [leftControl, rightControl];
            this.VerticalWeights = [this.VerticalWeights, 1];
        end
        
        function [sectionIndex, subSectionIndex] = findRow(this, rowID)
            sectionIndex = [];
            subSectionIndex = [];
            for i=1:numel(this.SectionStructs)
                sectionStruct = this.SectionStructs{i};
                j = find( strcmp({sectionStruct.RowID}, rowID), 1);
                if ~isempty(j)
                    sectionIndex = i;
                    subSectionIndex = j;
                    break; 
                end
            end
        end
    end
    
end

% helpers
function iAddSpacer(parent)
h = uicontrol('Parent', parent, 'Style', 'text', 'String', '', 'HorizontalAlignment', 'left');
h.Units = 'pixels';
h.Position(4) = 3;
end

function margin = iTinyMargin()
margin = 1e-9;
end

function pointsPerPixel = iGetPointsPerPixel()
pointsPerInch = 72;
pixelPerInch = get(groot, 'ScreenPixelsPerInch');
pointsPerPixel = pointsPerInch/pixelPerInch;
end

function height = iTextHeight()
height = 23;
end
