function impred = runRFL(srforest, imlow)
impred = applySRF(imlow,srforest,length(srforest.model),1);

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
