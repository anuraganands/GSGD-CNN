% Column strategy for printing verbose log of deepDreamImage.
%
% Copyright 2016-2017 The MathWorks, Inc.

classdef DeepDreamImageColumnStrategy < nnet.internal.cnn.util.ColumnStrategy
    
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
        function this = DeepDreamImageColumnStrategy()            
            this.Column(1).Name = "Iteration";
            this.Column(1).Type = "d";
            this.Column(1).Header = getString(message('nnet_cnn:deepDreamImage:Iteration'));
            this.Column(2).Name = "ActivationStrength";
            this.Column(2).Type = ".2f";
            this.Column(2).Header = getString(message('nnet_cnn:deepDreamImage:ActivationStregth'));
            this.Column(3).Name = "Octave";
            this.Column(3).Type = "d";
            this.Column(3).Header = getString(message('nnet_cnn:deepDreamImage:Octave'));
            
            rawHeaders = {this.Column.Header};
            this.TableHeaderManager = nnet.internal.cnn.util.ProgressDisplayTableHeader( ...
                rawHeaders );
            
            columnWidths = this.TableHeaderManager.ColumnWidths;
            this.Column = this.formatColumns( this.Column, columnWidths );
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