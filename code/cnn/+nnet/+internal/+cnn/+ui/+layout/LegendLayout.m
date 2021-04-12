classdef(Abstract) LegendLayout < handle
    % LegendLayout   Interface for legends with checkboxes
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Abstract)
        % Parent   (graphics handle) The parent component
        Parent 
        
        % AreCheckboxesVisible   (logical) Are all the checkboxes visible?
        AreCheckboxesVisible
    end
    
    properties(Abstract, SetAccess = private)
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
    
    methods(Abstract)
        addSection(this, sectionName, sectionStructArr)
        % addSection   Adds a section to the legend. The sectionName is the
        % section's title, and sectionStructArr defines the content of the
        % section. sectionStructArr should be a struct array with the
        % following fields:
        %   - LineStyle
        %   - LineWidth
        %   - LineColor
        %   - Marker
        %   - MarkerFaceColor
        %   - Text
    end
    
end

