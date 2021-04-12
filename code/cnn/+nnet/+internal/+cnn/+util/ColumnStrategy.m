classdef (Abstract) ColumnStrategy
    % ColumnStrategy   Column strategy interface
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties (Abstract, SetAccess=protected)
        % Column (struct)   Struct array of table columns
        Column

        % HorizontalBorder (char array)   Horizontal border of the table to
        %                                 print
        HorizontalBorder
        
        % Headings (char array)   Table headings
        Headings
    end
    
    properties (Abstract, Access=protected)
        % TableHeaderManager   An object of type nnet.internal.cnn.util.ProgressDisplayTableHeader
        TableHeaderManager
    end
    
    methods (Access=protected)
        function columns = formatColumns(~, columns, columnWidths)
            % formatColumns   Create column formatting operators from type
            % and width of columns. Populate Width and Format fields of
            % each column
            
            % Subtract 2 since each column is always padded with one space on each side
            columnWidths = columnWidths - 2;
            
            % Combine column width and type together
            for ii=1:numel(columns)
                columns(ii).Width = columnWidths(ii);
                columns(ii).Format = "%" + columnWidths(ii) + columns(ii).Type;
            end
        end
    end
end
