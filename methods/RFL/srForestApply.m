function out = srForestApply( dataLow, dataHigh, srforest, varargin )
% srForestApply Applies the trained super-resolution forest. 
% 
% This function applies the trained super-resolution forest [1] to a set of
% given images. If the high-resolution images are provided, an evaluation
% is also done. 
% 
% USAGE
%  opts = srForestApply( )
%  out  = srForestApply( dataLow, dataHigh, srforest, varargin )
% 
% INPUTS
%  dataLow      - low-res image or path to low-res images (.bmp). If
%                 dataLow = '', dataHigh is downscaled with the information 
%                 from srForestModel
%  dataHigh     - high-res image or path to high-res images (.bmp). If
%                 provided, the results are evaluated as well. 
%  srforest     - trained super-resolution forest (see srForestTrain.m)
%  varargin     - additional params (struct or name/value pairs)
%   .rmborder   - [3] #pixels to remove from border, for EVALUATION ONLY!
%   .Mhat       - [length(srforest.model)] number of trees used for 
%                 inference
%   .nthreads   - [1] number of threads used for forest inference
% 
% OUTPUTS
%  out          - struct array with output information for each image
%   .im         - upscaled image
%   .eval       - struct with evaluation results (available if
%                 high-resolution images are given)
%    .srf.psnr  - PSNR of the super-resolution forest
%    .bic.psnr  - PSNR of bicubic upsampling
% 
% See also: srForestTrain
%
% REFERENCES
% [1] S. Schulter, C. Leistner, H. Bischof. Fast and Accurate Image
%     Upscaling with Super-Resolution Forests. CVPR 2015.
% 
% The code is adapted from Piotr's Image&Video Toolbox (Copyright Piotr
% Dollar). 

dfs={ 'rmborder',3,  'Mhat',[],  'nthreads',1  };
opts=getPrmDflt(varargin,dfs,1); if nargin == 0, out=opts; return; end;

% check input
if isempty(dataLow)&&isempty(dataHigh), error('Either dataLow or dataHigh has to be provided'); end;
if isempty(srforest), error('the model srForest has to be provided'); end;
if isempty(opts.Mhat), opts.Mhat=length(srforest.model); end;

% check if we need to downscale the high-res images first for evaluation!
downscale = false; if isempty(dataLow), downscale=true; dataLow=dataHigh; end;

% iterate the low-res images and upscale them
if ischar(dataLow)
  imlistLow = dir(fullfile(dataLow,'*.bmp')); nimgs=length(imlistLow);
else
  nimgs=1;
end
if ischar(dataHigh), imlistHigh = dir(fullfile(dataHigh,'*.bmp')); end;
for i=1:nimgs
  if srforest.sropts.verbose, fprintf('Upscale image %d/%d\n',i,nimgs); end;
  
  % get the low-res image
  if ischar(dataLow), imL=imread(fullfile(dataLow,imlistLow(i).name));
  else imL=dataLow; end;
  [imLY,imLCB,imLCR] = imageTransformColor(imL);
  imLY = imageModcrop(imLY,srforest.sropts.sf);
  if ~isempty(imLCB)&&~isempty(imLCR)
    imLCB = imageModcrop(imLCB,srforest.sropts.sf);
    imLCR = imageModcrop(imLCR,srforest.sropts.sf);
  end
  if downscale
    imLY=imageDownsample(imLY,srforest.sropts.sf,srforest.sropts.downsample);
    if ~isempty(imLCB)&&~isempty(imLCR)
      imLCB=imageDownsample(imLCB,srforest.sropts.sf,srforest.sropts.downsample);
      imLCR=imageDownsample(imLCR,srforest.sropts.sf,srforest.sropts.downsample);
    end
  end;
  
  % bicubic upsampling of the Y channel to generate the mid-res image
  imMY = imresize(imLY,srforest.sropts.sf,srforest.sropts.interpkernel);
  
  % generate super-resolution forest output
  imHYPred = applySRF(imMY,srforest,opts.Mhat,opts.nthreads);
  
  % do some evaluations for SRF (& bicubic upsampling)
  if ischar(dataHigh)&&~isempty(dataHigh), imH=imread(fullfile(dataHigh,imlistHigh(i).name));
  else imH=dataHigh; end;
  if ~isempty(imH)
    % prepare GT image
    imHYGT = imageTransformColor(imH);
    imHYGT = imageModcrop(imHYGT,srforest.sropts.sf);
    
    % check if rmborder is enough!
    rmBorder = srforest.sropts.patchBorder + ...
      floor((srforest.sropts.patchSizeLow-srforest.sropts.patchSizeHigh)/2);
    imSize=size(imHYGT); imSize=imSize-2*rmBorder;
    rmBorder = rmBorder + mod(imSize-srforest.sropts.patchSizeHigh,...
      srforest.sropts.patchStride);
    if any(opts.rmborder<rmBorder), error('opts.rmBorder is set too small'); end;
    
    % remove border for evaluation
    imHYPredEval=cropBorder(imHYPred,opts.rmborder);
    imMYEval=cropBorder(imMY,opts.rmborder);
    imHYGTEval=cropBorder(imHYGT,opts.rmborder);

    % evaluate SRF & bicubic upsampling
    out(i).eval.srf = evaluateQuality(imHYGTEval,imHYPredEval);
    out(i).eval.bic = evaluateQuality(imHYGTEval,imMYEval);
  end
  
  % generate output image (RGB or grayscale) for SRF
  if ~isempty(imLCB)&&~isempty(imLCR)
    imHCB=imresize(imLCB,srforest.sropts.sf,srforest.sropts.interpkernel);
    imHCR=imresize(imLCR,srforest.sropts.sf,srforest.sropts.interpkernel);
    outimgYCRCB=cat(3,imHYPred,imHCB,imHCR); outimg=ycbcr2rgb(uint8(255*outimgYCRCB));
  else
    outimg = uint8(255*imHYPred);
  end; 
  out(i).im = outimg;
end
end

function imout = applySRF( imMY, srforest, Mhat, nthreads )
% set some constants
opts=srforest.sropts;
tarBorder = opts.patchBorder+floor((opts.patchSizeLow-opts.patchSizeHigh)/2);

% extract patches and compute features
patchesSrc = extractPatches(imMY,opts.patchSizeLow,...
  opts.patchSizeLow-opts.patchStride,opts.patchBorder,opts.patchfeats);
patchesSrcPca = srforest.Vpca' * patchesSrc;

% apply random regression forest
if opts.useARF
  patchesTarPred = forestAltRegrApply(patchesSrcPca,patchesSrcPca,...
    srforest.model,Mhat,nthreads);
  patchesTarPred = cat(3,patchesTarPred{:}); 
  patchesTarPred=sum(patchesTarPred,3)/size(patchesTarPred,3);
else
  patchesTarPred = forestRegrApply(patchesSrcPca,patchesSrcPca,...
    srforest.model,Mhat,nthreads);
  patchesTarPred = cat(3,patchesTarPred{:}); 
  patchesTarPred=sum(patchesTarPred,3)/size(patchesTarPred,3);
end

% add mid-res patches + patches predicted by SRF
patchesMid = extractPatches(imMY,opts.patchSizeHigh,...
  opts.patchSizeHigh-opts.patchStride,tarBorder);
patchesTarPred = patchesTarPred + patchesMid;

% merge patches into the final output (i.e., average overlapping patches)
img_size = size(imMY); patchSizeHigh = srforest.sropts.patchSizeHigh;
grid = getSamplingGrid(img_size,...
  patchSizeHigh,patchSizeHigh-opts.patchStride,tarBorder,1);
imout = overlap_add(patchesTarPred,img_size,grid);
end

function result = overlap_add( patches, img_size, grid )
% Image construction from overlapping patches
result = zeros(img_size,'single');
weight = zeros(img_size);
for i = 1:size(grid,3)
  patch = reshape(patches(:, i), size(grid, 1), size(grid, 2));
  result(grid(:, :, i)) = result(grid(:, :, i)) + patch;
  weight(grid(:, :, i)) = weight(grid(:, :, i)) + 1;
end
I = logical(weight);
result(I) = result(I) ./ weight(I);
end

function scores = evaluateQuality( imgt, impred )
% As done in R. Timofte, V. De Smet, L. van Gool. Anchored Neighborhood 
% Regression for Fast Example-Based Super- Resolution. ICCV 2013. 
scores.psnr=psnr(uint8(255*imgt),uint8(255*impred));
end

function imnoborder = cropBorder( im, rmBorder )
imnoborder = im(rmBorder+1:end-rmBorder,rmBorder+1:end-rmBorder,:);
end
