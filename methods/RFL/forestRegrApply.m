function XtarPred = forestRegrApply( Xfeat, Xsrc, forest, Mhat, NCores )
% forestRegrApply Applies a trained regression forest. 
% 
% Given the data, Xfeat (for splitting functions) and Xsrc (for leaf node
% predictions), this function applies the trained forest. 
% 
% The parameter Mhat defines the number of trees (out of the model) that
% are really evaluated. NCores defines the number of CPU cores that are
% used for parallelizing the inference procedure. 
% 
% USAGE
%  retdata = forestRegrApply( Xfeat, Xsrc, forest, Mhat, nthreads )
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
% See also forestRegrTrain, forestAltRegrTrain, forestAltRegrApply
%
% The code is adapted from Piotr's Image&Video Toolbox (Copyright Piotr
% Dollar). 

assert(isa(Xfeat,'single')); M=length(forest); nthreads = 1;
if nargin >= 4
  if (Mhat < 1 || Mhat > M), error('Mhat is set wrong: 0 < Mhat <= M'); end;
else
  Mhat = length(forest);
end
if nargin >= 5, nthreads = NCores; if NCores<1, error('NCores set below 1!'); end; end;
assert(~isempty(forest));

% get the leaf node prediction type and check with the given Xsrc data
[~,Fs]=size(forest(1).leafinfo(end).T);
leafpredtype = forest(1).leafinfo(end).type;
switch leafpredtype
  case 0
    assert(Fs==1);
  case 1
    assert(Fs==size(Xsrc,1)+1); % +1 for bias term!
    Xsrc = [Xsrc; ones(1,size(Xsrc,2),'single')];
  case 2
    assert(Fs==size(Xsrc,1)*2+1); % *2 + 1 for doubled size and bias term
    Xsrc = [Xsrc; ones(1,size(Xsrc,2),'single'); Xsrc.^2];
  otherwise
    error('Unknown leaf node prediction model');
end

% iterate the trees
myforest = forest(1:Mhat); node2leafids = cell(Mhat,1); treeleafs = cell(Mhat,1);
for i=1:length(myforest), tree=myforest(i);
  node2leafids{i} = zeros(size(tree.child),'uint32');
  node2leafids{i}(tree.child==0) = cat(1,tree.leafinfo(tree.child==0).id);
  treeleafs{i} = tree.leafinfo(tree.child==0);
  tree.fids = tree.fids - 1; myforest(i) = tree;
end
XtarPred = forestRegrInference(Xfeat',Xsrc',myforest,node2leafids,...
  treeleafs,leafpredtype,nthreads);
end

