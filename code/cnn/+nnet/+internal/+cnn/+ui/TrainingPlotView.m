classdef(Abstract) TrainingPlotView < handle
    % TrainingPlotView   Interface for training plot views
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Abstract, SetAccess = private)
        % Figure   (figure) The figure for the view.
        Figure
        
        % FigureExistsAndIsVisible   (logical) Is the main figure visible
        % and existent?
        FigureExistsAndIsVisible
    end
    
    properties(Abstract)
        % ProgressPanelVisible   (logical) Is the panel showing all the
        % progress information visible?
        ProgressPanelVisible
        
        % TrainingErrorMessageVisible    (logical) Is the message saying
        % that training errored out visible?
        TrainingErrorMessageVisible
        
        % PlotErrorMessageVisible    (logical) Is the message saying that
        % the plot errored out visible?
        PlotErrorMessageVisible
    end
    
    events
        % HelpLinkClicked  Event fired when user clicks on help link 
        HelpLinkClicked
        
        % StopButtonClicked   Event fired when user clicks on the stop
        % button
        StopButtonClicked
        
        % FigureCloseRequested   Event fired when user clicks the close
        % button
        FigureCloseRequested
    end
    
    methods(Abstract)        
        closeFigure(this)
        % closeFigure   Close the figure
        
        setUserData(this, userData)
        % setUserData   Sets user data.
        
        setTitleInfo(this, trainingStartTime)
        % setTitleInfo   Sets the information required to produce a title
        % for the figure
        
        showMainPlot(this)
        % showMainPlot   Shows the main plot
        
        
        setTopAxes(this, axesView)
        % setTopAxes   Sets top axes
        
        setBottomAxes(this, axesView)
        % setBottomAxes   Sets bottom axes
        
        
        setupProgress(this, currValue, maxValue)
        % setupProgress   Setup the progress info in the view
        
        updateProgress(this, currValue)
        % updateProgress   Update the progress info in the view
        
        
        setupLegend(this, cellOfLegendSectionNames, cellOfLegendSectionStructArrs, checkboxesVisible)
        % setupLegend   Setup the legend with multiple sections. Each
        % section comes with a section name and a struct array. Each struct
        % array must contain the following fields:
        %   - LineStyle
        %   - LineWidth
        %   - LineColor
        %   - Marker
        %   - MarkerFaceColor
        %   - Text
        
        
        setupTableOfData(this, cellOfTableSectionNames, cellOfTableSectionStructArrs)
        % setupTableOfData   Setup the table of data. Each section comes
        % with a section name and a struct array. Each struct array must
        % contain the following fields:
        %  - RowID
        %  - LeftText
        %  - RightText
        
        updateTableOfData(this, rowID, newRightText)
        % updateTableOfData   Update the table of data. The (right) text on
        % the given row will be updated with the given newRightText.
    end
end

