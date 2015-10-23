function patches = extractPatches( im, patchsize, overlap, border, varargin )
% extractPatches Extracts patches from an image. 
% 
% Given an image im (single channel) and settings about the patch 
% extraction (patchsize, overlap, and border), this function extracts all 
% possible patches. 
% 
% Moreover, additional options allow for computing features on the patches
% that are based on simple filter responses. The filters can also be
% defined. 
% 
% INPUTS
%  im           - [imh x imw] an image with a single channel
%  patchsize    - [ph x pw] size of the patches to be extracted
%  overlap      - [oy x ox] overlap of neighboring patches. Maximum overlap
%                 is [ph x pw] - [1 x 1]
%  border       - [by x bx] border of images that is left out
%  varargin     - additional params (struct or name/value pairs)
%   .type       - ['none'] what kind of features to compute. A second
%                 option is 'filters', where also the .filters options
%                 should be set. 
%   .filters    - [] filter kernels (for .type='filters'). This should be a
%                 a cell array with different filters. 
% 
% OUTPUTS
%  patches      - [ph*pw*numel(.filters) x npatches] extracted patches
%                 (vectorized). npatches defines the number of patches that
%                 could be extracted with the given settings. 
% 
% See also getSamplingGrid
% 
% Code adapted from [1]
% 
% References:
% [1] R. Timofte, V. De Smet, L. van Gool. Anchored Neighborhood Regression 
% for Fast Example-Based Super- Resolution. ICCV 2013. 

dfs={ 'type','none',  'filters',[]  };
opts=getPrmDflt(varargin,dfs,1); if nargin == 0, patches=opts; return; end;

% check input
if size(im,3)~=1, error('im should have only a single channel'); end;
if any(patchsize<3), error('patchsize shoud be >= 3'); end;

% Compute one grid for all filters
grid = getSamplingGrid(size(im),patchsize,overlap,border,1);
switch opts.type
  case 'none'
    f = im(grid); patches=reshape(f,[size(f,1)*size(f,2),size(f,3)]);
  case 'filters'
    feature_size = prod(patchsize)*numel(opts.filters);
    patches = zeros([feature_size,size(grid,3)],'single');
    for i = 1:numel(opts.filters)
        f = conv2(im,opts.filters{i},'same'); f=f(grid);
        f = reshape(f,[size(f,1)*size(f,2),size(f,3)]);
        patches((1:size(f,1))+(i-1)*size(f,1),:) = f;
    end
  otherwise
    error('Unknown features');
end
end