function srforest = srForestTrain( varargin )
% srForestTrain Training of a super-resolution random forest
% 
% This function trains a random regression forest for the task of single
% image super-resolution as described in [1]. In general, one has to define
% the path to the training data. Either only the high-resolution images are
% given, which are then downsampled, or both high- and low-resolution
% images are given. This code only works with .bmp image files. 
% 
% Then, the upscaling factor sf has to be defined, as well as all settings
% about the patch size, stride, etc. One can also define the number of
% training samples, and what kind of features are computed. 
% 
% Finally, the settings for the random forest (or the alternating random
% forest [2]) has to be defined. 
% 
% USAGE
%  opts     = srForestTrain( )
%  srforest = srForestTrain( varargin )
% 
% INPUTS
%  varargin               - additional params (struct or name/value pairs)
%   .datapathHigh         - ['REQ'] path to high-dim train images (.bmp!)
%   .datapathLow          - [''] path to low-dim train images (.bmp!) If
%                           this is set empty, the high-resolution images
%                           are downscaled and used as low-resolution. 
%   .sf                   - [2] upscaling factor
%   .downsample           - [..] downsample params (see imageDownsample.m)
%   .patchSizeLow         - [6 6] patch size in low-dim images
%   .patchSizeHigh        - [6 6] patch size in high-dim images
%   .patchStride          - [2 2] stride of patch extraction
%   .patchBorder          - [2 2] border of patch extraction
%   .nTrainPatches        - [0] #train-patches in total (if 0, all are
%                           used)
%   .nAddBaseScales       - [0] number of additional base scales to extract
%                           training patches from as is done in A+ [3]. 
%   .patchfeats           - [..] feature params (see computeFeats.m)
%   .interpkernel         - ['bicubic'] interpolation kernel for later
%                           upscaling (used in srForestApply)
%   .pRegrForest          - [..] forest params (see forestRegrTrain.m)
%   .useARF               - [0] use alternating regression forest
%   .verbose              - [1] verbosity level
% 
% OUTPUTS
%  srforest               - super-resolution forest
%   .model                - the random regression forest model trained
%   .Vpca                 - PCA projection matrix for feature computation
%   .sropts               - the options struct
% 
% See also: srForestApply, forestRegrTrain, forestAltRegrTrain
% 
% REFERENCES:
% [1] S. Schulter, C. Leistner, H. Bischof. Fast and Accurate Image
%     Upscaling with Super-Resolution Forests. CVPR 2015.
% [2] S. Schulter, C. Leistner, P. Wohlhart, P. M. Roth, H. Bischof. 
%     Alternating Regression Forests for Object Detection and Pose 
%     Estimation. ICCV 2013. 
% [3] R. Timofte, V. De Smet, L. van Gool. A+: Adjusted Anchored
%     Neighborhood Regression for Fast Super-Resolution. ACCV 2014. 
% 
% The code is adapted from Piotr's Image&Video Toolbox (Copyright Piotr
% Dollar). 

dfs={ 'datapathHigh','REQ',  'datapathLow','',  'sf',2,  ...
  'downsample',[],  'patchSizeLow',[6 6],  'patchSizeHigh',[6 6],  ...
  'patchStride',[2 2], 'patchBorder',[2 2], ...
  'nTrainPatches',0,  'nAddBaseScales',0,  'patchfeats',[],  ...
  'interpkernel','bicubic',  'pRegrForest',forestRegrTrain(),  ...
  'useARF',0,  'verbose',1  };
opts=getPrmDflt(varargin,dfs,1); if nargin == 0, srforest=opts; return; end;

% check some parameter settings
if opts.sf<2, error('sf should be >= 2'); end;
if any(opts.patchSizeLow<3), error('patchSizeLow should be >= 3'); end;
if any(opts.patchSizeHigh<3), error('patchSizeHigh should be >= 3'); end;
if opts.nTrainPatches<0, error('nTrainPatches should be >= 0'); end;
if any(opts.patchStride>opts.patchSizeHigh), 
  error('Stride is too large for high-res patch size'); end;

% extract source and target patches
imlistHigh = dir(fullfile(opts.datapathHigh,'*.bmp')); imlistLow = [];
if ~isempty(opts.datapathLow)
  imlistLow = dir(fullfile(opts.datapathLow,'*.bmp'));
  if length(imlistHigh)~=length(imlistLow), error('#low-res ~= #high-res images'); end;
end;
patchesSrcCell = cell(1,length(imlistHigh)); patchesTarCell=patchesSrcCell;
nPatchesPerImg = 0;
if opts.nTrainPatches>0, nPatchesPerImg = floor(opts.nTrainPatches/length(imlistHigh)); end;
for i=1:length(imlistHigh)
  if opts.verbose, fprintf('Extract patches from image %d/%d\n',i,length(imlistHigh)); end;
  
  % read low- and high-res images
  imH = imread(fullfile(opts.datapathHigh,imlistHigh(i).name)); imL = [];
  if ~isempty(imlistLow)
    imL=imread(fullfile(opts.datapathLow,imlistLow(i).name));
  end;
  
  % process images (ycbcr color space, etc.) and crop patches
  baseScales = .98.^(0:opts.nAddBaseScales);
  [patchesSrcCell{i},patchesTarCell{i}] = extractPatchesFromImg(imH,imL,opts,baseScales);
  
  % control the number of extracted patches
  if nPatchesPerImg>0
    cNumPatches = size(patchesSrcCell{i},2);
    patchesSrcCell{i} = patchesSrcCell{i}(:,1:min(cNumPatches,nPatchesPerImg));
    patchesTarCell{i} = patchesTarCell{i}(:,1:min(cNumPatches,nPatchesPerImg));
  end
end
patchesSrc = cat(2,patchesSrcCell{:}); clear patchesSrcCell;
patchesTar = cat(2,patchesTarCell{:}); clear patchesTarCell;
if opts.verbose, fprintf('Extracted a total of %d patches from %d images\n',...
    size(patchesSrc,2),length(imlistHigh)); end;
opts.pRegrForest.N1=size(patchesSrc,2);

% reduce dimensionality of low-res patches (PCA)
if opts.verbose, fprintf('Applying PCA dim-reduction\n'); end;
C = double(patchesSrc*patchesSrc');
[V,D] = eig(C); D = diag(D); D = cumsum(D) / sum(D);
k = find(D >= 1e-3,1); % ignore 0.1% energy
srforest.Vpca = V(:,k:end); % choose the largest eigenvectors' projection
clear C D V;
fprintf('\t %d to %d dimensions\n',size(srforest.Vpca,1),size(srforest.Vpca,2));
patchesSrcPca = srforest.Vpca' * patchesSrc;

% train the regression forest
if opts.useARF
  if opts.verbose, fprintf('Training alternating regression forest\n'); end;
  srforest.model = forestAltRegrTrain(patchesSrcPca,patchesSrcPca,...
    patchesTar,opts.pRegrForest);
else
  if opts.verbose, fprintf('Training regression forest\n'); end;
  srforest.model = forestRegrTrain(patchesSrcPca,patchesSrcPca,...
    patchesTar,opts.pRegrForest);
end
srforest.sropts = opts;
end

function [srcPatches,tarPatches] = extractPatchesFromImg(imH,imL,opts,baseScales)
if nargin<4, baseScales=1; end; srcPatches=[]; tarPatches=[];
% iterate all scales
for i=1:length(baseScales)

  % down sample image(s)
  imH = imageDownsample(imH,baseScales(i),opts.downsample);
  if ~isempty(imL), imL=imageDownsample(imL,baseScales(i),opts.downsample); end;
  
  % process images (ycbcr color space, etc.) and crop patches
  imHY = imageTransformColor(imH); % we only need the Y channel!
  imHY = imageModcrop(imHY,opts.sf);
  if isempty(imL)
    imLY = imageDownsample(imHY,opts.sf,opts.downsample);
  else
    imLY = imageTransformColor(imL); % we only need the Y channel!
  end

  % compute midres image (this is also what happens during inference!)
  imMY = imresize(imLY,opts.sf,opts.interpkernel);

  % compute the target image (i.e., the patches we want to regress!)
  imTar = imHY - imMY;

  % extract corresponding patches
  srcPatches = [srcPatches,extractPatches(imMY,opts.patchSizeLow,...
    opts.patchSizeLow-opts.patchStride,opts.patchBorder,opts.patchfeats)];
  tarBorder = opts.patchBorder + floor((opts.patchSizeLow-opts.patchSizeHigh)/2);
  tarPatches = [tarPatches,extractPatches(imTar,opts.patchSizeHigh,...
    opts.patchSizeHigh-opts.patchStride,tarBorder)]; % no features
end
end
