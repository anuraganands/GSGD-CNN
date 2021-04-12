classdef ClassificationColumns < nnet.internal.cnn.util.ColumnStrategy
    % ClassificationColumns   Classification column strategy
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties (Access = protected)
        % TableHeaderManager   An object of type nnet.internal.cnn.util.ProgressDisplayTableHeader
        TableHeaderManager
    end
    
    properties (SetAccess=protected)
        % Column (struct)   Struct array of table columns
        Column
    end
    
    properties (SetAccess=protected, Dependent)
        % HorizontalBorder (char array)   Horizontal border of the table to
        %                                 print
        HorizontalBorder
        
        % Headings (char array)   Table headings
        Headings
    end
    
    methods
        function this = ClassificationColumns()
            this.Column(1).Name = "Epoch";
            this.Column(1).Type = "d";
            this.Column(1).Header = getString(message('nnet_cnn:internal:cnn:ClassificationColumns:Epoch'));
            this.Column(2).Name = "Iteration";
            this.Column(2).Type = "d";
            this.Column(2).Header = getString(message('nnet_cnn:internal:cnn:ClassificationColumns:Iteration'));
            this.Column(3).Name = "Time";
            this.Column(3).Type = "s";
            this.Column(3).Header = getString(message('nnet_cnn:internal:cnn:ClassificationColumns:Time'));
            this.Column(4).Name = "Accuracy";
            this.Column(4).Type = ".2f%%";
            this.Column(4).Header = getString(message('nnet_cnn:internal:cnn:ClassificationColumns:Accuracy'));
            this.Column(5).Name = "Loss";
            this.Column(5).Type = ".4f";
            this.Column(5).Header = getString(message('nnet_cnn:internal:cnn:ClassificationColumns:Loss'));            
            this.Column(6).Name = "LearnRate";
            this.Column(6).Type = ".4f";
            this.Column(6).Header = getString(message('nnet_cnn:internal:cnn:ClassificationColumns:LearnRate'));
            
            rawHeaders = {this.Column.Header};
            this.TableHeaderManager = nnet.internal.cnn.util.ProgressDisplayTableHeader( ...
                rawHeaders );
            
            columnWidths = this.TableHeaderManager.ColumnWidths;
            
            % Adjust accuracy width to take the extra % into account
            columnWidths(4) = columnWidths(4) - 1;
            
            % Format columns
            this.Column = this.formatColumns( this.Column, columnWidths );
            
            % Undo the width adjustment, so that the real width is stored
            this.Column(4).Width = this.Column(4).Width + 1;
        end
    end
    
    methods
        function horizontalBorder = get.HorizontalBorder(this)
            horizontalBorder = this.TableHeaderManager.HorizontalBorder;
        end
        
        function headings = get.Headings(this)
            headings = this.TableHeaderManager.Header;
        end
    end
end