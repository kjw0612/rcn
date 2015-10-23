function XtarPred = forestAltRegrApply( Xfeat, Xsrc, forest, Mhat, NCores )
% forestAltRegrApply Applies a trained alternating regression forest. 
% 
% Given the data, Xfeat (for splitting functions) and Xsrc (for leaf node
% predictions), this function applies the trained alternating regression
% forest [1]. 
% 
% The parameter Mhat defines the number of trees (out of the model) that
% are really evaluated. NCores defines the number of CPU cores that are
% used for parallelizing the inference procedure. 
% 
% Please note that inference in alternating regression forests [1] and
% standard regression random forests is exactly the same. Thus, this
% function directly calls forestRegrApply. The only reason for having a
% separate function is to avoid any confusion. 
% 
% USAGE
%  retdata = forestAltRegrApply( Xfeat, Xsrc, forest, Mhat, nthreads )
%
% INPUTS
%  Xfeat    - [FfxN] N length Ff feature vectors
%  Xsrc     - [FsxN] N length Fs data vectors
%  forest   - learned regression forest model
%  Mhat     - [length(forest)] number of trees used for inference
%  NCores   - #CPU cores that should be used for inference
%
% OUTPUTS
%  XtarPred - [FtxN] N length Ft predicted target vectors
%
% See also forestAltRegrTrain, forestRegrTrain, forestRegrApply
%
% REFERENCES
% [1] S. Schulter, C. Leistner, P. Wohlhart, P. M. Roth, H. Bischof. 
%     Alternating Regression Forests for Object Detection and Pose 
%     Estimation. ICCV 2013. 
% 
% The code is adapted from Piotr's Image&Video Toolbox (Copyright Piotr
% Dollar). 

if nargin < 4, Mhat=length(forest); end;
if nargin < 5, NCores=1; end;
XtarPred = forestRegrApply( Xfeat, Xsrc, forest, Mhat, NCores );
