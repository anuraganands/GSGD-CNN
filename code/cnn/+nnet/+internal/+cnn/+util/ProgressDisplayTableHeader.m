classdef ProgressDisplayTableHeader
    % ProgressDisplayTableHeader   Class to create progress display table
    % header
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Constant)
        % PreferredMaxLineLength   Maximum preferred line length. This can
        % be bigger if the first word is bigger. Sentences that are bigger
        % than this quantity will be split in multiple lines
        PreferredMaxLineLength = 14
        
        % ColumnPadding   How much padding to put around each header
        ColumnPadding = 2
    end
    
    properties
        % Header   The header for the table (does not include horizontal
        % borders)
        Header
        
        % ColumnWidths   Width of each of the N columns, expressed as a 1xN
        % vector
        ColumnWidths
        
        % HorizontalBorder   An horizontal border with the same length as
        % the header
        HorizontalBorder
    end
    
    methods
        function this = ProgressDisplayTableHeader( rawHeaders )
            % ProgressDisplayTableHeader   Construct a
            % ProgressDisplayTableHeader from a cell array of headers
            % rawHeaders
            [this.Header, headerLength, this.ColumnWidths] = iGetHeader( rawHeaders, ...
                this.PreferredMaxLineLength, this.ColumnPadding );
            this.HorizontalBorder = iCreateHorizontalBorder( headerLength );
        end
    end
end

function [header, headingsLength, columnWidths] = iGetHeader( rawHeaders, minimumHeaderWordLength, columnPadding )
% iGetHeader   Transform a char array of raw headings sapareted by | into
% display-ready headings, well spaced and wrapped into multiple lines when
% needed

% Wrap headers on multiple lines
% wrappedHeaders = arrayfun(@(s)iWrapColumnHeader(s, minimumHeaderWordLength), rawHeaders, 'UniformOutput', false);
wrappedHeaders = cellfun(@(s)iWrapColumnHeader(s, minimumHeaderWordLength), rawHeaders, 'UniformOutput', false);

% Ge the maximum number of rows for the header. If an header does not have
% as many rows, fill them with blank strings
headerRows = max(cellfun(@numel,wrappedHeaders));

% Create the final header by merging the single headers with separators and
% padding and by adding blank strings when needed
columnPaddingString = iCreateBlankString( columnPadding );
header = "";
numHeaders = numel(wrappedHeaders);
headingsLength = [];
for row=1:headerRows
    for col=1:numHeaders
        currentHeaderDepth = numel(wrappedHeaders{col});
        if currentHeaderDepth<row
            currentColumnWidth = wrappedHeaders{col}(1).strlength;
            blankString = iCreateBlankString( currentColumnWidth );
            currentHeading = blankString;
        else
            currentHeading = wrappedHeaders{col}(row);
        end
        header = sprintf("%s|%s%s%s",header,columnPaddingString,currentHeading,columnPaddingString);
    end
    if isempty(headingsLength)
        headingsLength = strlength(header) - 1;
    end
    header = sprintf("%s|\n",header);
end

% Strip out last \n
header = header.strip;

% Compute column widths
columnWidths = cellfun(@(s)strlength(s(1)),wrappedHeaders);
columnWidths = columnWidths'+strlength(columnPaddingString)*2;
end

function wrappedTextOutput = iWrapColumnHeader( inputText, minimumStringLength )
% iWrapColumnHeader   Split text into multiple lines. Return an array of
% strings (one string per line), where all strings are of the same length
% (at least minimumStringLength). Text is centrally aligned

wrappedText = matlab.internal.display.printWrapped(inputText, minimumStringLength);
wrappedText = string(wrappedText);
wrappedTextSplit = split(wrappedText, newline);

% Delete the last line, being only composed of a newline symbol
wrappedTextSplit(end) = [];

% Centrally align text
wrappedTextOutput = wrappedTextSplit.pad('both');
end

function horizontalBorder = iCreateHorizontalBorder(headerLength)
% iCreateHorizontalBorder   Create an horizontal border of length headerLength
horizontalBorder = "|" + string(repmat('=',1,headerLength)) + "|";
end

function blankString = iCreateBlankString( strLength )
% iCreateBlankString   Create a blank string of length strLength
blankString = string(repmat(' ',1,strLength));
end