function forest = forestAltRegrTrain( Xfeat, Xsrc, Xtar, varargin )
% forestAltRegrTrain Train an alternating regression forest. 
% 
% We have a regression problem where we want to map from Xsrc to Xtar. For
% more flexibility, we separate the data for splitting the data (in the
% tree) and making the predictions in the leaf nodes. Thus, 
% 
% Xfeat = \Phi(Xsrc)
% 
% is simply some feature representation of Xsrc. Xfeat will be used in the
% splitting functions, while Xsrc will be used in the leaf nodes for, e.g.,
% a linear prediction model (Xtar = P * Xsrc). For a constant leaf node
% model, this is irrelevant. If also the leaf node models should use Xfeat
% for prediction, just call the function with (Xfeat, Xfeat, Xtar, ...). 
%
% This method implements the algorithm from [1], i.e., alternating
% regression forests that optimize a global loss over all trees. 
%
% Dimensions:
%  M    - number trees
%  N    - number input vectors
%  Ff   - dimensionality of Xfeat
%  Fs   - dimensionality of Xsrc
%  Ft   - dimensionality of Xtar
%
% USAGE
%  opts   = forestAltRegrTrain( )
%  forest = forestAltRegrTrain( Xfeat, Xsrc, Xtar, [varargin] )
%
% INPUTS
%  Xfeat      - [REQ: Ff x N] N length Ff feature vectors
%  Xsrc       - [REQ: Fs x N] N length Fs source vectors
%  Xtar       - [REQ: Ft x N] N length Ft target vectors
%  varargin   - additional params (struct or name/value pairs)
%   .M        - [1] number of trees to train
%   .minCount - [128] minimum number of data points to allow split
%   .minChild - [64] minimum number of data points allowed at child nodes
%   .N1       - [N*.75] number of data points for training each tree
%   .F1       - [sqrt(F)] number features to sample for each node split
%   .F2       - [5] number of thresholds to sample for each node split. If
%               F2=0, the median of the responses is chosen. 
%   .maxDepth - [64] maximum depth of tree
%   .fWts     - [] weights used for sampling features
%   .splitfuntype - ['pair'] split function type: single or pair tests
%   .nodesubsample - [1000] data subsampling on the node level. 0 means no
%               subsampling is done. Values > 0 indicate the size of the 
%               subsample
%   .splitevaltype - ['variance'] Type of split function evaluation:
%               three options are available: 'balanced', 'variance',
%               'reconstruction'
%   .lambda   - [.01] regularization parameter least squares problems
%               (splitting and leafs)
%   .estimatelambda [0] - try to estimate lambda automatically
%   .kappa     - [1] regularization parameter for split quality
%   .leaflearntype - ['linear'] dictionary learning variant for the leaf
%                    nodes: constant, linear, polynomial
%   .ARFloss  - ['squared'] regression loss for ARF. Options are 'squared'
%   .ARFlambda- [1.0] regularization parameter (shrinkage) for ARF
%   .usepf    - [0] use parfor for training trees
%   .verbose  - [0] verbosity level (only 0 and 1 available)
%
% OUTPUTS
%  forest   - learned forest model struct array with the following fields
%   .fids     - [Kx(1 or 2)] feature ids for each node
%   .thrs     - [Kx1] threshold corresponding to each fid
%   .child    - [Kx1] index of child for each node
%   .count    - [Kx1] number of data points at each node
%   .depth    - [Kx1] depth of each node
%   .leafinfo - [Kx1] leaf node information
%     .T      - [Ftx(1 or Fs)] regression matrix
%     .type   - [0/1/2] constant or linear/poynomial prediction model
%     .id     - leaf node id
%
% See also: forestAltRegrApply, forestRegrTrain
%
% REFERENCES
% [1] S. Schulter, C. Leistner, P. Wohlhart, P. M. Roth, H. Bischof. 
%     Alternating Regression Forests for Object Detection and Pose 
%     Estimation. ICCV 2013. 
% 
% The code is adapted from Piotr's Image&Video Toolbox (Copyright Piotr
% Dollar). 

% get additional parameters and fill in remaining parameters
dfs={ 'M',1,  'minChild',64,  'minCount',128,  'N1',[],  'F1',[],  ...
  'F2',5,  'maxDepth',64,  'fWts',[],  'splitfuntype','pair',  ...
  'nodesubsample',1000,  'splitevaltype','variance', ...
  'lambda',0.01,  'estimatelambda',0,  'kappa',1,  'leaflearntype','linear', ...
  'ARFloss','squared',  'ARFlambda',1.0,  'usepf',0,  'verbose',0 };
opts=getPrmDflt(varargin,dfs,1); if nargin == 0, forest = opts; return; end;
[Ff,N]=size(Xfeat); [~,Ncheck]=size(Xsrc); assert(N==Ncheck); [~,Ncheck]=size(Xtar);
assert(N==Ncheck);
opts.minChild=max(1,opts.minChild); opts.minCount=max([1 opts.minCount opts.minChild]);
if(isempty(opts.N1)), opts.N1=round(N*.75); end; opts.N1=min(N,opts.N1);
if(isempty(opts.F1)), opts.F1=round(sqrt(Ff)); end; opts.F1=min(Ff,opts.F1);
if(opts.F2<0), error('F2 should be > -1'); end;
if(isempty(opts.fWts)), opts.fWts=ones(1,Ff,'single'); end; opts.fWts=opts.fWts/sum(opts.fWts);
if(opts.nodesubsample<2*opts.minChild), error('nodesubsample < 2*minChild'); end;
if(opts.nodesubsample<3*opts.minChild), warning('nodesubsample < 3*minChild'); end;

% make sure data has correct types
if(~isa(Xfeat,'single')), Xfeat=single(Xfeat); end
if(~isa(Xsrc,'single')), Xsrc=single(Xsrc); end
if(~isa(Xtar,'single')), Xtar=single(Xtar); end
if(~isa(opts.fWts,'single')), opts.fWts=single(opts.fWts); end

% algorithm start ...
cPred = zeros(size(Xtar),'single'); tmp_forests=cell(opts.M,1);
dWtsUni = ones(1,N,'single'); dWtsUni=dWtsUni/sum(dWtsUni);
baggingIndices=zeros(opts.M,opts.N1,'uint32');
for i=1:opts.M
  if N==opts.N1, baggingIndices(i,:)=1:N; 
  else baggingIndices(i,:)=wswor(dWtsUni,opts.N1,4); end;
end
for j=1:(opts.maxDepth+2) % to be compatible with forestRegrTrain()!
  if opts.verbose, fprintf('Training level %03d of %03d\n',j,opts.maxDepth); end;
  
  % compute pseudo targets
  cPseudoTar = computePseudoTargets(Xtar,cPred,opts.ARFloss);
  
  % train M random trees on different subsets of data
  if opts.usepf
    parfor i=1:opts.M
      if N==opts.N1 %#ok<PFBNS>
        tmp_forests{i}=treeRegrTrain(tmp_forests{i},j-1,Xfeat,Xsrc,cPseudoTar,opts);
      else
        d=baggingIndices(i,:); Xfeat1=Xfeat(:,d); Xsrc1=Xsrc(:,d); Ptar1=cPseudoTar(:,d);
        tmp_forests{i}=treeRegrTrain(tmp_forests{i},j-1,Xfeat1,Xsrc1,Ptar1,opts);
      end
    end
  else
    for i=1:opts.M
      if N==opts.N1
        tmp_forests{i} = treeRegrTrain(tmp_forests{i},j-1,Xfeat,Xsrc,cPseudoTar,opts);
      else
        d=baggingIndices(i,:);
        Xfeat1=Xfeat(:,d); Xsrc1=Xsrc(:,d); Ptar1=cPseudoTar(:,d);
        tmp_forests{i} = treeRegrTrain(tmp_forests{i},j-1,Xfeat1,Xsrc1,Ptar1,opts);
      end
    end
  end
  for i=1:opts.M
    if(i==1), tmp=tmp_forests{i}; forest=tmp(ones(opts.M,1)); 
    else forest(i)=tmp_forests{i}; end;
  end
  
  % compute current prediction
  cPred = forestAltRegrApply(Xfeat,Xsrc,forest); 
  cPred = mean(cat(3,cPred{:}),3);
  
end
% clean-up the trees (rm dids)
for i=1:length(forest), forest(i).dids=[]; end;
end



% =========================================================================
% ========= helper function ===============================================

function tree = treeRegrTrain( tree, cdepth, Xfeat, Xsrc, Xtar, opts )
% Train a single regression tree

% define some constants
[~,N]=size(Xfeat); 
fidsDim=1; if strcmp(opts.splitfuntype,'pair'), fidsDim=2; end
% frequence for printing a status message (if in verbose mode)
msgnodestep = 200;

% train the tree
if isempty(tree)
  
  % initialize the tree with a single leaf node
  tree=struct('fids',zeros(1,fidsDim,'uint32'),'thrs',zeros(1,1,'single'),...
    'child',zeros(1,1,'uint32'),'count',N*ones(1,1,'uint32'),...
    'depth',zeros(1,1,'uint32'),'leafinfo',[],'dids',{uint32(1:N)});
  tree.leafinfo = createLeaf(Xsrc,Xtar,opts.leaflearntype,...
    opts.lambda,opts.estimatelambda);
  tree.leafinfo.id = 1;
  
else
  
  % find nodes in in previous level of depth
  cnodes = find(tree.depth==(cdepth-1));
  
  % current counter for new nodes
  K = length(tree.child) + 1;
  
  % increase tree structure
  nadd = length(cnodes)*2;
  tree.fids=[tree.fids;zeros(nadd,fidsDim,'uint32')];
  tree.thrs=[tree.thrs;zeros(nadd,1,'single')];
  tree.child=[tree.child;zeros(nadd,1,'uint32')];
  tree.count=[tree.count;zeros(nadd,1,'uint32')];
  tree.depth=[tree.depth;zeros(nadd,1,'uint32')];
  tree.dids=[tree.dids;cell(nadd,1)];
  
  for k=cnodes(:)'
    
    dids1=tree.dids{k}; XfeatNode=Xfeat(:,dids1);
    XsrcNode = Xsrc(:,dids1); XtarNode = Xtar(:,dids1);
    if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), ...
        fprintf('Node %04d, depth %02d, %07d samples () ',k,tree.depth(k),tree.count(k)); end
    % if insufficient data or max-depth reached, don't train split
    if( tree.count(k)<=opts.minCount||tree.depth(k)>opts.maxDepth||tree.count(k)<(2*opts.minChild) )
      if (opts.verbose && (mod(k,msgnodestep)==0)||k==1), ...
          fprintf('stays a leaf (stop criterion active)\n'); end
      continue;
    end
    if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), fprintf('find split () '); end
    
    %
    % turn previous leaf into split node!
    %
    
    % compute responses for all data samples
    switch opts.splitfuntype
      case 'single'
        fids1=wswor(opts.fWts,opts.F1,4); resp=XfeatNode(fids1,:);
      case 'pair'
        fids1 = [wswor(opts.fWts,opts.F1,4); wswor(opts.fWts,opts.F1,4)];
        % Caution: same feature id could be sampled  -> all zero responses
        resp=XfeatNode(fids1(1,:),:)-XfeatNode(fids1(2,:),:);
      otherwise
        error('Unknown splitfunction type');
    end

    % subsample the data for splitfunction node optimization
    if opts.nodesubsample > 0 && opts.nodesubsample < tree.count(k)
      randinds = randsample(tree.count(k),opts.nodesubsample);
      respSub = resp(:,randinds); XsrcSub = XsrcNode(:,randinds);
      XtarSub = XtarNode(:,randinds);
    else
      respSub = resp; XsrcSub = XsrcNode; XtarSub = XtarNode;
    end
    if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), ...
        fprintf('subsmpl = %07d/%07d () ',size(respSub,2),size(resp,2)); end

    % find best splitting function and corresponding threshold
    [fid,thr,rerr]=findSplitAndThresh(respSub,XsrcSub,XtarSub,opts.F2,...
      opts.splitevaltype,opts.lambda,opts.minChild,opts.kappa);

    % check validity of the splitting function
    validsplit=false;
    left=resp(fid,:)<thr; count0=nnz(left); fid=fids1(:,fid);
    if ~isinf(rerr) && count0>=opts.minChild && (tree.count(k)-count0)>=opts.minChild
      validsplit=true;
    end

    % continue tree training (either split or create a leaf)
    if validsplit
      % set info for current node
      tree.child(k)=K; tree.fids(k,:)=fid; tree.thrs(k)=thr;
      % set info for child nodes
      tree.dids{K}=dids1(left); tree.dids{K+1}=dids1(~left); 
      tree.depth(K:K+1)=tree.depth(k)+1; 
      tree.count(K)=length(dids1(left)); tree.count(K+1)=length(dids1(~left));
      % to save memory, we can delete the dids for the split node. 
      tree.dids{k}=[];
      % we have to create the leaf nodes already here!
      parentT = tree.leafinfo(k).T;
      % (LEFT)
      XsrcChild = Xsrc(:,tree.dids{K}); XtarChild = Xtar(:,tree.dids{K});
      tree.leafinfo(K,1) = createLeaf(XsrcChild,XtarChild,opts.leaflearntype,...
        opts.lambda,opts.estimatelambda);
      tree.leafinfo(K,1).T = opts.ARFlambda*parentT + tree.leafinfo(K,1).T;
      if any(isnan(tree.leafinfo(K,1).T(:))), error('Found NaN value in leaf'); end;
      % (RIGHT)
      XsrcChild = Xsrc(:,tree.dids{K+1}); XtarChild = Xtar(:,tree.dids{K+1});
      tree.leafinfo(K+1,1) = createLeaf(XsrcChild,XtarChild,opts.leaflearntype,...
        opts.lambda,opts.estimatelambda);
      tree.leafinfo(K+1,1).T = opts.ARFlambda*parentT + tree.leafinfo(K+1,1).T;
      if any(isnan(tree.leafinfo(K+1,1).T(:))), error('Found NaN value in leaf'); end;
      % clear parent leaf node info, it became a split node
      tree.leafinfo(k).T=[];
      tree.leafinfo(k).type=-1;
      tree.leafinfo(k).id=-1;
      % increase counter
      K=K+2;
      
      if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), ...
          fprintf('valid split (loss=%.6f)\n',rerr); end;
    else
      if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), ...
          fprintf('invalid split -> stays a leaf\n'); end;
    end
    
  end; 
  
  % remove not used, but pre-allocated nodes
  K=K-1;
  tree.fids=tree.fids(1:K,:);
  tree.thrs=tree.thrs(1:K);
  tree.child=tree.child(1:K);
  tree.count=tree.count(1:K);
  tree.depth=tree.depth(1:K);
  tree.dids=tree.dids(1:K);
  
  % create the leaf-node id mapping
  leafcnt = 0;
  for i=1:length(tree.leafinfo)
    if ~isempty(tree.leafinfo(i).T), leafcnt=leafcnt+1; ...
        tree.leafinfo(i).id = leafcnt; end;
  end
  
end
end

function [fid, thr, rerr] = findSplitAndThresh( resp, Xsrc, Xtar, F2, splitevaltype, lambda, minChild, kappa )
[F1,~]=size(resp); rerr=Inf; fid=1; thr=Inf; Ft=size(Xtar,1); Fs=size(Xsrc,1);
% special treatment for random tree growing
if strcmp(splitevaltype,'random'), F1=1; F2=1; end;
% iterate the random split functions
for s=1:F1
  % get thresholds to evaluate
  if F2==0, tthrs=median(resp(s,:));
  else
    respmin=min(resp(s,:)); respmax=max(resp(s,:));
    tthrs = zeros(F2+1,1,'single'); % we also add the median as threshold
    tthrs(1:end-1) = rand(F2,1)*0.95*(respmax-respmin) + respmin;
    tthrs(end) = median(resp(s,:));
  end
  % iterate the thresholds
  for t=1:length(tthrs)
    tthr=tthrs(t); left=resp(s,:)<tthr; right=~left; 
    nl=nnz(left); nr=nnz(right);
    if nl<minChild || nr<minChild, continue; end;
    XsrcL=Xsrc(:,left); XsrcR=Xsrc(:,right);
    XtarL=Xtar(:,left); XtarR=Xtar(:,right);
    % compute the quality if the splitting function
    switch splitevaltype
      case 'random'
        trerr = 0; % this is better than Inf (it can be constant because we only evaluate once)
      case 'balanced'
        trerr = (nl - nr)^2;
      case 'variance'
        trerrL = sum(var(XtarL,0,2))/Ft; trerrR = sum(var(XtarR,0,2))/Ft;
        if kappa>0
          trerrLsrc=sum(var(XsrcL,0,2))/Fs; trerrRsrc=sum(var(XsrcR,0,2))/Fs;
          trerrL=(trerrL+kappa*trerrLsrc)/2; trerrR=(trerrR+kappa*trerrRsrc)/2;
        end
        trerr = (nl*trerrL + nr*trerrR)/(nl+nr);
      case 'reconstruction' % based on a sampled dictionary
        XsrcL = [XsrcL; ones(1,size(XsrcL,2),'single')]; %#ok<AGROW>
        TL =  XtarL * ((XsrcL*XsrcL' + lambda*eye(size(XsrcL,1))) \ XsrcL)';
        XsrcR = [XsrcR; ones(1,size(XsrcR,2),'single')]; %#ok<AGROW>
        TR =  XtarR * ((XsrcR*XsrcR' + lambda*eye(size(XsrcR,1))) \ XsrcR)';
        trerrL = sqrt(sum(sum((XtarL-TL*XsrcL).^2))/nl);
        trerrR = sqrt(sum(sum((XtarR-TR*XsrcR).^2))/nr);
        if kappa > 0% regularizer
          trerrLsrc=sum(var(XsrcL,0,2))/Fs; trerrRsrc=sum(var(XsrcR,0,2))/Fs;
          trerrL=(trerrL+kappa*trerrLsrc)/2; trerrR=(trerrR+kappa*trerrRsrc)/2;
        end
        trerr = (nl*trerrL + nr*trerrR) / (nl+nr);
      otherwise
        error('Unknown split evaluation type');
    end

    if trerr<rerr, rerr=trerr; thr=tthr; fid=s; end;
  end
end

end

function leaf = createLeaf( Xsrc, Xtar, leaflearntype, lambda, autolambda )
% creates a leaf node and computes the prediction model

% downsample the data a bit
max_num_samples = 5000; ndata = size(Xsrc,2);
if ndata>max_num_samples
  randinds=randperm(ndata); randinds=randinds(1:max_num_samples);
  Xsrc=Xsrc(:,randinds); Xtar=Xtar(:,randinds);
end

switch leaflearntype
  case 'constant'
    T = sum(Xtar,2)/size(Xtar,2); predmodeltype = 0;
  case 'linear'
    Xsrc = [Xsrc; ones(1,size(Xsrc,2),'single')];
    matinv = Xsrc*Xsrc'; if autolambda, lambda=estimateLambda(matinv); end
    T = Xtar * ((matinv + lambda*eye(size(Xsrc,1))) \ Xsrc)';
    predmodeltype = 1;
  case 'polynomial' % actually, it is only polynomial with quadratic term :)
    Xsrc = [Xsrc; ones(1,size(Xsrc,2),'single'); Xsrc.^2];
    matinv = Xsrc*Xsrc'; if autolambda, lambda=estimateLambda(matinv); end
    T = Xtar * ((matinv + lambda*eye(size(Xsrc,1))) \ Xsrc)';
    predmodeltype = 2;
  otherwise
    error('Unknown leaf node prediction type');
end
leaf.T=T; leaf.type=predmodeltype; leaf.id=-1;
end

function lambda = estimateLambda(matinv)
rcondTmpMat = rcond(matinv); if rcondTmpMat<eps, rcondTmpMat = 1e-10; end;
lambda = 0;
if rcondTmpMat < 1e-2, lambda = 10^(-4 - log10(rcondTmpMat)) * rcondTmpMat; end
if isnan(lambda), error('oida, des suit goa nit passiern!'); end;
end

function ids = wswor( prob, N, trials )
% Fast weighted sample without replacement. Alternative to:
%  ids=datasample(1:length(prob),N,'weights',prob,'replace',false);
M=length(prob); assert(N<=M); if(N==M), ids=1:N; return; end
if(all(prob(1)==prob)), ids=randperm(M); ids=ids(1:N); return; end
cumprob=min([0 cumsum(prob)],1); assert(abs(cumprob(end)-1)<.01);
cumprob(end)=1; [~,ids]=histc(rand(N*trials,1),cumprob);
[s,ord]=sort(ids); K(ord)=[1; diff(s)]~=0; ids=ids(K);
if(length(ids)<N), ids=wswor(cumprob,N,trials*2); end
ids=ids(1:N)';
end

function G = computePseudoTargets( Y, F, loss )
switch loss
  case 'squared'
    G = Y - F;
  otherwise
    error('Unknown ARF loss');
end
end