classdef NetworkAnalyzer < handle
    % NetworkAnalyzer   Analyze Neural Networks to obtain diagnostic
    %                   information.
    %
    % To create a NetworkAnalyzer use analyzeNetwork
    %   
    % Properties:
    %   LayerAnalyzers      Array of LayerAnalyzers associated with the
    %                       layers in the network.
    %
    %   ExternalLayers      The external layers in the network.
    %   InternalLayers      The internal layers in the network.
    %
    %   Connections         The connections table (based on layer names and
    %                       port names).
    %   InternalConnections The internal connections matrix (based on
    %                       indexes).
    %   HiddenConnections   The hidden connections table (based on 
    %                       indexes).
    %
    %   LayerGraph          The layer graph associated with the network
    %                       analyzed.
    %   IsSeriesNetwork     Logical value indicating whether the network
    %                       can be represented as a SeriesNetwork.
    %
    % Methods:
    %   this = NetworkAnalyzer(input)
    %       Constructs a NetworkAnalyzer from the input. The input can be:
    %        * A SeriesNetwork
    %        * A DAGNetwork
    %        * An array of nnet.cnn.layer.Layer
    %        * A nnet.cnn.LayerGraph
    %
    %   this.applyConstraints(constraints)
    %       Applies the constrain object "constraints" to the
    %       NetworkAnalyzer. This should populates the Errors and Warnings
    %       in the LayerAnalyzesrs.
    %       If no constraitns are given, all the buit-in constraints are
    %       applied.
    %
    %   this.throwIssuesIfAny()
    %       Takes the contraints from LayerIssues, and produces a warning
    %       listign all detected warnings, and throws an error with all the
    %       detected errors.
    %       This method would be tipically called after calling the method
    %       applyConstraints.
    %       
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties ( SetAccess = private )
        
        LayerAnalyzers(:,1) nnet.internal.cnn.analyzer.util.LayerAnalyzer;
        Connections table;
        
        Issues table;

    end
    
    properties ( Dependent, SetAccess = private )
        
        ExternalLayers(:,1) nnet.cnn.layer.Layer;
        LayerGraph(1,1) nnet.cnn.LayerGraph;
        IsSeriesNetwork(1,1) logical;
        
    end
    
    properties ( Dependent, SetAccess = private, Hidden )
        
        InternalLayers(:,1) cell;
        
        InternalConnections(:,4) double;
        HiddenConnections table;
        
    end
    
    properties ( Access = private )
        
        TopologicalOrder(:,1) double;
        
    end
    
    methods
        
        function v = get.ExternalLayers(this)
            v = [this.LayerAnalyzers.ExternalLayer];
            v = reshape(v, size(this.LayerAnalyzers));
        end
        function set.ExternalLayers(this, v)
            this.LayerAnalyzers = arrayfun(@iLayerAnalyzer, v);
            this.LayerAnalyzers = iAddLayerIndex(this.LayerAnalyzers);
            this.LayerAnalyzers = iDeduceNames(this.LayerAnalyzers);
        end
        function lgraph = get.LayerGraph(this)
            layers = [this.LayerAnalyzers.ExternalLayer]';
            conn = this.HiddenConnections;
            
            sizes = {this.LayerAnalyzers.InternalOutputSizes}';
            topoOrder = this.TopologicalOrder';
            
            lgraph = nnet.cnn.LayerGraph(layers, conn);
            lgraph = lgraph.setSizes(sizes);
            lgraph = lgraph.setTopologicalOrder(topoOrder);
        end
        function tf = get.IsSeriesNetwork(this)
            conn = this.InternalConnections;
            n = numel(this.LayerAnalyzers);
            tf = size(conn,1) == n ...
                && all(conn(:,1) == (1:n-1)') ...
                && all(conn(:,2) == 1) ...
                && all(conn(:,3) == (2:n)') ...
                && all(conn(:,4) == 1);
        end
        
        function v = get.InternalLayers(this)
            v = {this.LayerAnalyzers.InternalLayer}';
        end
        
        function set.Connections(this, v)
            this.Connections = v;
            
            % We have a set of connections, make sure the layers are sorted
            % topologically according to these connections.
            i = pseudotoposort(this.InternalConnections(:,[1 3]), ...
                               numel(this.LayerAnalyzers)); %#ok<MCSUP>
            this.TopologicalOrder = i; %#ok<MCSUP>
            this.LayerAnalyzers = this.LayerAnalyzers(i(:)); %#ok<MCSUP>
        end
        function v = get.InternalConnections(this)
            v = nnet.internal.cnn.util.externalToInternalConnections( ...
                    this.Connections, this.ExternalLayers);
        end
        function set.InternalConnections(this, v)
            this.Connections = nnet.internal.cnn.util.internalToExternalConnections( ...
                    v, this.ExternalLayers);
        end
        function v = get.HiddenConnections(this)
            v = nnet.internal.cnn.util.internalToHiddenConnections( ...
                    this.InternalConnections);
        end
        
    end
    
    methods
        
        function this = NetworkAnalyzer(input)
            [layers, connections] = iExtractSortedLayers(input);
            
            this.ExternalLayers = layers;
            this.InternalConnections = connections;
            this.propagateSizes();
        end
        
        function applyConstraints(this, constraints)
            if nargin < 2
                constraints = nnet.internal.cnn.analyzer...
                    .constraints.Constraint.getBuiltInConstraints();
            end
            
            assert(...
                isa(constraints, "nnet.internal.cnn.analyzer.constraints.Constraint"), ...
                iMessage('NetworkAnalyzer:InvalidConstraint') );
            
            this.Issues = [
                this.Issues
                constraints.applyConstraints(this) ];
        end
        
        function throwIssuesIfAny(this)
            issues = this.Issues;
            
            if isempty(issues)
                return;
            end
            
            errors = issues(issues.Severity == "E", :);
            warnings = issues(issues.Severity == "W", :);
            
            % Warnings contain information like layer renaming. Show warnings first.
            if ~isempty(warnings)
                exception = iBuildException( ...
                    'NetworkAnalyzer:NetworkHasWarnings', warnings);
                iExceptionAsWarnignsWithoutBacktrace(exception);
            end            
            if ~isempty(errors)
                exception = iBuildException( ...
                    'NetworkAnalyzer:NetworkHasErrors', errors);
                throwAsCaller(exception);
            end
        end
        
    end
    
    methods ( Access = private )
        
        function propagateSizes(this)
            % Do the size propagation through the network.
            
            connections = this.Connections;
            internalConnections = this.InternalConnections;
            
            layer = internalConnections(:, [1 3]);
            ports = internalConnections(:, [2 4]);
            
            % Go through each layer (they are topologically sorted)
            for i = 1:numel(this.LayerAnalyzers)
                inputs = ( layer(:,2) == i );
                
                % Changing the input table triggers the input validation
                % for the layer, so avoid doing many minor changes; make a
                % copy of the table, and then update all at once.
                inputTable = this.LayerAnalyzers(i).Inputs;
                
                % Go through each input to the layer, and connect it to its
                % source (as long as it is a valid connection).
                % Non valid connections can arise from SeriesNetworks where
                % a layer with no output can be placed before other layers.
                for j = find(inputs')
                    outputTable = this.LayerAnalyzers(layer(j,1)).Outputs;
                    
                    % Fill up the source/destination information
                    if ~isempty(inputTable)
                        inputTable.Source{ports(j,2)}(end+1) = ...
                                string(connections.Source{j});
                    end
                    if ~isempty(outputTable)
                        outputTable.Destination{ports(j,1)}(end+1) = ...
                                string(connections.Destination{j});
                    end
                    
                    % Fill up the sizes of the connection
                    if ~isempty(outputTable) && ~isempty(inputTable)
                        inputTable.Size{ports(j,2)} = ...
                            outputTable.Size{ports(j,1)};
                    end
                    
                    this.LayerAnalyzers(layer(j,1)).Outputs = outputTable;
                end
                
                % Update the input table.
                this.LayerAnalyzers(i).Inputs = inputTable;
            end
        end
        
    end
    
end

function la = iLayerAnalyzer(varargin)
    la = nnet.internal.cnn.analyzer.util.LayerAnalyzer(varargin{:});
end

function out = iHiddenToInernalConnections(in)
    out = nnet.internal.cnn.util.hiddenToInternalConnections(in);
end

function [layers, internalConnections] = iExtractSortedLayers(input)
    % Extract the layers and connections from either:
    %  * A SeriesNetwork
    %  * A DAGNetwork
    %  * An array of nnet.cnn.layer.Layer
    %  * A nnet.cnn.LayerGraph
    %
    % For arrays of layers or series networks, the connections are
    % automatically generated.
    % For layer graphs or DAG networks, the layers are topologically
    % sorted (doing a best effort if cycles are present).
    
    if isa(input, 'SeriesNetwork')
        input = input.Layers;
    elseif isa(input, 'DAGNetwork')
        input = layerGraph(input);
    end
    
    if isa(input,'nnet.cnn.layer.Layer')
        layers = input;
        n = numel(layers);
        internalConnections = [1:(n-1); ones(1,n-1); 2:n; ones(1,n-1)]';
    elseif isa(input,'nnet.cnn.LayerGraph')
        layers = input.Layers;
        internalConnections = ...
            iHiddenToInernalConnections(input.HiddenConnections);
    else
        error(iMessage('NetworkAnalyzer:InvalidInput'));
    end
    
    if isempty(layers)
        error(iMessage('NetworkAnalyzer:EmptyNetwork'));
    end
end

function layerAnalyzers = iAddLayerIndex(layerAnalyzers)
    n = numel(layerAnalyzers);
    ind = mat2cell((1:n)',ones(n,1));
    [layerAnalyzers.OriginalIndex] = ind{:};
end

function layerAnalyzers = iDeduceNames(layerAnalyzers)
    % Deduce the name of the layers, replacing empty names with
    % their default values, and making sure there are no
    % duplicates.

    current = string({layerAnalyzers.Name});
    
    namesAreDirty = ~isempty(iGetDuplicatedNames([current ""]));
    if ~namesAreDirty
        return;
    end
    
    original = string({layerAnalyzers.OriginalName});
    default = string({layerAnalyzers.DefaultName});

    hasName = (original ~= "");
    
    % Infer layer names for layers with a predefiend name
    names(hasName)  = iRenameDuplicated(original(hasName));
    names(~hasName) = iRenameDuplicated( ...
        default(~hasName), [names(hasName) original(hasName)]);

    % Set the layers name using the deduced names
    for i=1:numel(layerAnalyzers)
        layerAnalyzers(i).Name = char(names(i));
    end
end

function duplicated = iGetDuplicatedNames(names)
    % Generate list of duplicated names
    [~,i] = unique(names);
    i = setdiff(1:numel(names), i);
    duplicated = unique(names(i));
end

function renamed = iRenameDuplicated(names, avoid)
    % Makes a list of unique names, avoiding using the names that were
    % duplicated and the names specidied in 'avoid'
    if nargin < 2
        avoid = string.empty();
    end
    dup = iGetDuplicatedNames(names);
    renamed = matlab.lang.makeUniqueStrings( ...
        names, [avoid(:); dup(:)]);
end

function i = pseudotoposort(st, n)
    % Try to figure out a topological order for a directed graph.
    % If the graph id a DAG, then the topological order is returned.
    % If the graph is not a DAG, edges are removed until it is a DAG, and
    % the topological order of the resulting graph is returned.
    
    st = unique(st, 'rows');
    g = digraph(st(:,1), st(:,2), [], n);
    
    dag = dgraph2dag(g);
    i = toposort(dag, 'Order', 'stable');
end

function gdag = dgraph2dag(g)
    % Converts a directed graph into a directed acyclic graph by removing
    % edges.

    M = adjacency(g);
    numLayers = size(M,1);
    
    % Locate input nodes (nodes with no inputs)
    in = find(indegree(g) == 0);
    if isempty(in)
        in = 1; % The graph is a huge cycle, there is no clear input node
    end

    % Make gdag acyclic by removing cycle-edges found by dfsearch starting
    % at the inputs
    M(numLayers+1, numLayers+1) = 0; % add a helper node
    M(numLayers+1, in) = 1;   % with edges to all sources

    restart = true; % needed if there are unconnected cyclic graphs
    edges = dfsearch(digraph(M),numLayers+1,'edgetodiscovered','Restart',restart);

    M = M(1:numLayers, 1:numLayers); % remove helper node
    
    % Remove all cycle-edges
    M(sub2ind(size(M), edges(:, 1), edges(:, 2))) = 0;

    gdag = digraph(M);
end

function msg = iMessage(id, varargin)
    id = string(id);
    newId = "nnet_cnn:internal:cnn:analyzer:" + id;
    msg = message(newId{1}, varargin{:});
    
    try
        string(msg);
    catch
        msg = message(id{1}, varargin{:});
    end
end

function exception = iBuildException(messageId, issues)
    exception = MException(iMessage(messageId));

    layerIssues = issues(issues.Type == "Layer", :);
    networkIssues = issues(issues.Type == "Network", :);

    [msgNetwork, idNetwork] = iGetNetworkIssuesMessage(networkIssues);
    [msgLayer, idLayer] = iGetLayerIssuesMessage(layerIssues);
    
    msg = [msgNetwork; msgLayer];
    id = [idNetwork; idLayer];
    userData = [networkIssues.UserData; layerIssues.UserData];
    for i=1:numel(msg)
        cause = MException(id{i}, msg{i});
        if isfield(userData{i}, 'cause')
            for j=1:numel(userData{i}.cause)
                cause = cause.addCause(userData{i}.cause{j});
            end
        end
        exception = exception.addCause( cause );
    end
end

function [msg, id] = iGetNetworkIssuesMessage(issues)
    id = strrep(string(issues.Id), ":internal:", ":");
    msg = strings(size(issues.Message));
    for i=1:numel(msg)
        header = string(iMessage('NetworkAnalyzer:NetworkIssueHeader'));
        msg(i) = string(iMessage('NetworkAnalyzer:IssueMessage', ...
            iBold(header), issues.Message{i}));
    end
end

function [msg, id] = iGetLayerIssuesMessage(issues)
    id = strrep(string(issues.Id), ":internal:", ":");
    msg = strings(size(issues.Message));
    for i=1:numel(msg)
        header = string(iMessage('NetworkAnalyzer:LayerIssueHeader', ...
            issues.LayerDisplayNames{i}));
        msg(i) = string(iMessage('NetworkAnalyzer:IssueMessage', ...
            iBold(header), issues.Message{i}));
    end
end

function iExceptionAsWarnignsWithoutBacktrace(exception)
    backtrace = warning('query','backtrace');
    warning('off','backtrace');
    warning(exception.identifier, ""+exception.getReport()+newline);
    warning(backtrace.state,'backtrace');
end

function str = iBold(str)
    % Add markup to make text bold
    if matlab.internal.display.isHot()
        str = "<strong>" + str + "</strong>";
    end
end