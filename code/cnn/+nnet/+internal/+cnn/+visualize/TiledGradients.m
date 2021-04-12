classdef TiledGradients
    % TiledGradients   Methods to compute tiled gradients for visualization
    %
    % Compute gradients in tiles to reduce memory usage. Split each
    % gradient computation up into smaller 'tiles' so that no one
    % computation is too large. Then recombine the tiles to give the
    % original gradient.
    
    %   Copyright 2016 The MathWorks, Inc.
    
    methods (Static)
        
        function [tiledGradient, activations] = computeTiledGradient(iVisualNet, X, tileSideLength)
            % tiledGradient   Compute gradients in tiles to reduce memory
            % usage. Split each gradient computation up into smaller
            % 'tiles' so that no one computation is too large. Then
            % recombine the tiles to give the original gradient. iVisualNet
            % is a Visual Network. X is an image for which to compute the
            % derivative of the output of the visual network.
            % tileSideLength is the size of a tile.
    
            height = size(X, 1);
            width = size(X, 2);

            if height > tileSideLength || width > tileSideLength
                
                % Roll X
                [rollX, rollY] = nnet.internal.cnn.visualize.TiledGradients.rollAmount(...
                    X, tileSideLength);
                X = circshift(X, [rollY, rollX]);
                
                % Pad X with zeros
                X = nnet.internal.cnn.visualize.TiledGradients.padToCompatibleTileSize(...
                    X, tileSideLength);
                
                % Compute the tiles
                tiles = nnet.internal.cnn.visualize.TiledGradients.splitIntoTiles(X, tileSideLength);
                
                numTilesVertical = size(tiles, 5);
                numTilesHorizontal = size(tiles, 6);
                
                gradientTiles = zeros(size(tiles), 'like', tiles);
                activationsPerChannel = zeros([numTilesVertical numTilesHorizontal size(X,4)], 'like', tiles);
                % Compute the gradient for each tile.
                for i=1:numTilesVertical
                    for j=1:numTilesHorizontal
                        [gradientTiles(:,:,:,:,i,j), activationsPerChannel(i,j,:)] = ...
                            nnet.internal.cnn.visualize.Gradients.computeGradient(...
                            iVisualNet, tiles(:,:,:,:,i,j));
                    end
                end
                
                % Put tiles back together into 4D array
                tiledGradient = nnet.internal.cnn.visualize.TiledGradients.reassembleTiles(gradientTiles);
                
                % Only keep the part of gradient without padding
                tiledGradient = tiledGradient(1:height, 1:width, 1:size(X,3), 1:size(X,4));
                
                % Unroll the gradient
                tiledGradient = circshift(tiledGradient, [-rollY, -rollX]);
                
            else
                % Do not tile if X is smaller than tileSideLength.
                [tiledGradient, activationsPerChannel] = ...
                    nnet.internal.cnn.visualize.Gradients.computeGradient(...
                    iVisualNet, X);
            end
            
            % Mean channel activation. Return channel activations as
            % [1 1 numChannels] array.
            activations = mean(mean(activationsPerChannel));
        end
        
        function paddedX = padToCompatibleTileSize(X, tileSideLength)
            % tileSideLength need not divide size(X, 1) or size(X, 2)
            % exactly. This function pads X with zeros at the end of the
            % first and second dimensions to make sure tileSideLength does
            % divide size(X, 1) and size(X, 2).
            
            % Compute the number of pixels to add vertically and
            % horizontally to get to a multiple of tileSideLength.
            padVertical = mod(tileSideLength - mod(size(X, 1), tileSideLength), ...
                tileSideLength);
            padHorizontal = mod(tileSideLength - mod(size(X, 2), tileSideLength), ...
                tileSideLength);
            
            % Pad X with zeros by padVertical and padHorizontal.
            paddedX = iPadWithZeros(X, [padVertical, padHorizontal]);                         
        end
        
        function tiles = splitIntoTiles(X, tileSideLength)
            % Splits a 4D array X into tiles of size tileSideLength x
            % tileSideLength x size(X, 3) x size(X, 4). Returns an array of
            % size tileSideLength x tileSideLength x size(X, 3) x size(X,
            % 4) x numTilesVertical x numTilesHorizontal.
            % It must be that tileSideLength divides size(X, 1) and size(X,
            % 2).
            
            % Create the tiles array
            numTilesVertical = size(X, 1) / tileSideLength;
            numTilesHorizontal = size(X, 2) / tileSideLength;
            tiles = zeros(tileSideLength, tileSideLength, size(X, 3), size(X, 4), ...
                numTilesVertical, numTilesHorizontal, 'like', X);
            
            % Put the tiles of X into the tiles array
            for i=1:numTilesVertical
                for j=1:numTilesHorizontal
                    startI = 1 + (i-1) * tileSideLength;
                    endI = startI + tileSideLength - 1;
                    startJ = 1 + (j-1) * tileSideLength;
                    endJ = startJ + tileSideLength - 1;
                    tiles(:,:,:,:,i,j) = X(startI:endI,startJ:endJ,:,:);
                end
            end
        end
        
        function assembled = reassembleTiles(tiles)
            % tiles a 6D array with the 5th and 6th dimensions index the
            % tiles. The first four dimensions correspond to the four
            % dimensions of the original 4D array. Reassemble the tiles
            % into a 4D array.
            
            numTilesVertical = size(tiles, 5);
            numTilesHorizontal = size(tiles, 6);
            
            tileSideLength = size(tiles, 1);
            
            assembled = zeros(size(tiles, 1) * numTilesVertical, ...
                size(tiles, 2) * numTilesHorizontal, ...
                size(tiles, 3), size(tiles, 4), 'like', tiles);
            
            for i=1:numTilesVertical
                for j=1:numTilesHorizontal
                    startI = 1 + (i-1) * tileSideLength;
                    endI = startI + tileSideLength - 1;
                    startJ = 1 + (j-1) * tileSideLength;
                    endJ = startJ + tileSideLength - 1;
                    assembled(startI:endI,startJ:endJ,:,:) = tiles(:,:,:,:,i,j);
                end
            end
        end
        
        function [rollX, rollY] = rollAmount(X, tileSize)
            % Compute the amount to roll X by. We only roll if the tile
            % size is less than size(X, 1) or size(X, 2).
            if (tileSize < size(X, 1)) || (tileSize < size(X, 2))
                rollX = randi(tileSize);
                rollY = randi(tileSize);
            else
                rollX = 0;
                rollY = 0;
            end
        end
    end
end

function Y = iPadWithZeros(X, padSize)
 % Pad X with zeros by padVertical and padHorizontal.
 %           
 % Equivalent to padarray(X, [padVertical, padHorizontal, 0,0], 0, 'post')
 
 assert(numel(padSize) == 2);
  
 padSize = [padSize repelem(0,ndims(X)-2)];
 
 Y = zeros(size(X) + padSize, 'like', X);
 
 % only pad first 2 dims
 Y(1:end-padSize(1), 1:end-padSize(2), :, :) = X;
 
end