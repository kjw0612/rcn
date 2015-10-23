function grid = getSamplingGrid( img_size, patch_size, overlap, border, scale )
% getSamplingGrid returns a grid to easily extract patches from an image
% 
% Given the size of an image, the window size (of the patches), the overlap
% of the patches, and the amount of border that is left out, this function
% computes a grid that can be easily used to extract all these patches from
% an image. The extraction of the patches then looks like 
% 
% im_patches = im(grid);
%
% The additional scale factor scales the window, the overlap, and the
% border accordingly. This might be useful for some applications. 
% 
% The output grid is of dimension [window(1) x window(2) x npatches], where
% npatches is that can be extracted from the image with these settings. 
% 
% INPUTS
%  img_size     - [imh x imw] size of the image to extract patches from
%  patch_size   - [ph x pw] size of the patches to be extracted
%  overlap      - [oy x ox] overlap of neighboring patches. Maximum overlap
%                 is [ph x pw] - [1 x 1]
%  border       - [by x bx] border of image that is left out
%  scale        - a scaling factor for patch_size, overlap, and border
% 
% OUTPUTS
%  grid         - [ph x pw x npatches] the sampling grid to easily extract
%                 patches from an image im, via: patches = im(grid);
%                 npatches is the number of patches that could be extracted
% 
% See also: extractPatches
% 
% Code adapted from [1]
% 
% References:
% [1] R. Timofte, V. De Smet, L. van Gool. Anchored Neighborhood Regression 
% for Fast Example-Based Super- Resolution. ICCV 2013. 

if nargin < 5, scale = 1; end;
if nargin < 4, border = [0 0]; end;
if nargin < 3, overlap = [0 0]; end;

% Scale all grid parameters
patch_size = patch_size * scale;
overlap = overlap * scale;
border = border * scale;

% Create sampling grid for overlapping window
index = reshape(1:prod(img_size), img_size);
grid = index(1:patch_size(1), 1:patch_size(2)) - 1;

% Compute offsets for grid's displacement.
skip = patch_size - overlap; % for small overlaps
offset = index(1+border(1):skip(1):img_size(1)-patch_size(1)+1-border(1), ...
               1+border(2):skip(2):img_size(2)-patch_size(2)+1-border(2));
offset = reshape(offset, [1 1 numel(offset)]);

% Prepare 3D grid - should be used as: sampled_img = img(grid);
grid = repmat(grid, [1 1 numel(offset)]) + repmat(offset, [patch_size 1]);
end