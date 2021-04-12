classdef SeriesLineModel < nnet.internal.cnn.ui.axes.LineModel
    % SeriesLineModel   Model for a line component
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties(AbortSet)
        % Name   (char) Name of line
        Name = ''
        
        % LineStyle   (char) Style of the line. One of '-' (default), 
        % '--' (dashed) or ':' (dotted)
        LineStyle = '-'
        
        % LineWidth   (double) Width of the line. Default is 1.3.
        LineWidth = 1.3
        
        % LineColor   (rgba-quadraplet) Color of the line e.g. [1,1,1,0.5]
        % is semi-translucent white. Default is [0,0,0,1]
        LineColor = [0,0,0,1]
        
        % MarkerType   (char) Type of marker. 'o' or 'none' (default).
        MarkerType = 'none'
        
        % MarkerSize   (double) Size of marker. Default is 5
        MarkerSize = 5
        
        % MarkerFaceColor   (rgb-triplet) Color of the marker face. Default
        % is [0,0,0]
        MarkerFaceColor = [0,0,0]
        
        % MarkerEdgeColor   (rgb-triplet) Color of the marker edge. Default
        % is [0,0,0]
        MarkerEdgeColor = [0,0,0]
        
        % HasFinalPointAnnotation (boolean) True if the final point should
        % be annotated.
        HasFinalPointAnnotation = false;
    end
    
    properties(Dependent, SetAccess = private)
        % XValues   (1xn doubles) The x-values of the line
        XValues
        
        % YValues   (1xn doubles) The y-values of the line
        YValues
        
        % XBounds   (1x2 doubles) The bounds on the x-values of the line
        % expressed as [min, max]
        XBounds
        
        % YBounds   (1x2 doubles) The bounds on the y-values of the line
        % expressed as [min, max]
        YBounds
        
        % MarkerIndices   (array of indices) Indices of XValues/YValues
        % which should be adorned with markers.
        MarkerIndices
    end
    
    properties(Access = private)
        % Series   (nnet.internal.cnn.ui.axes.Series) The series associated
        % with this line.
        Series
        
        % MarkerStrategy   (net.internal.cnn.ui.axes.MarkerStrategy)
        % Used to compute the MarkerIndices
        MarkerStrategy
    end
    
    methods
        function this = SeriesLineModel(series, markerStrategy)            
            this.Series = series;
            this.MarkerStrategy = markerStrategy;
        end
        
        function xValues = get.XValues(this)
            xValues = this.Series.XValues; 
        end
        
        function yValues = get.YValues(this)
            yValues = this.Series.YValues; 
        end
        
        function xbounds = get.XBounds(this)
            xbounds = this.Series.XBounds;
        end
        
        function ybounds = get.YBounds(this)
            ybounds = this.Series.YBounds;
        end
        
        function markerIndices = get.MarkerIndices(this)
            markerIndices = this.MarkerStrategy.computeMarkerIndices(this.Series.XValues); 
        end
    end
    
end

