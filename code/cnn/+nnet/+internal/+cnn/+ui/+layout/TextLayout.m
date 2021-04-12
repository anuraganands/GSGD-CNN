classdef(Abstract) TextLayout < handle
    % TextLayout   Interface for laying out of text
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Abstract)
        % Parent   (graphics handle) The parent component
        Parent 
    end
    
    properties(Abstract, SetAccess = private)        
        % MainPanel   (matlab.ui.container.Panel)
        % The main panel representing the entire legend.
        MainPanel
        
        % PreferredHeight   (integer) The preferred height of the MainPanel
        PreferredHeight
    end
    
    methods(Abstract)
        reset(this)
        % reset   Removes all sections
        
        addSection(this, sectionName, sectionStructArr) 
        % addSection   Add a section to the layout. The sectionName is the
        % section's title and the sectionStructArr is a struct array with
        % fields 'RowID', 'LeftText' and 'RightText'
        
        update(this, rowID, newRightText)
        % update   For the row with the given rowID, update the right text
        % with the newRightText.
    end
    
end

