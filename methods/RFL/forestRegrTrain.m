function forest = forestRegrTrain( Xfeat, Xsrc, Xtar, varargin )
% forestRegrTrain Trains a random regression forest. 
% 
% We have a regression problem where we want to map from Xsrc to Xtar. For
% more flexibility, we separate the data for splitting in the tree (Xfeat) 
% and making the predictions in the leaf nodes (Xsrc). Thus, 
% 
% Xfeat = \Phi(Xsrc)
% 
% is some feature representation of Xsrc. Xfeat will be used in the
% splitting functions, while Xsrc will be used in the leaf nodes for, e.g.,
% a linear prediction model (Xtar = P * Xsrc). For a constant leaf node
% model, this is irrelevant. If also the leaf node models should use Xfeat
% for prediction, just call the function with (Xfeat, Xfeat, Xtar, ...). 
%
% Dimensions:
%  M    - number trees
%  N    - number input vectors
%  Ff   - dimensionality of Xfeat
%  Fs   - dimensionality of Xsrc
%  Ft   - dimensionality of Xtar
%
% USAGE
%  opts   = forestRegrTrain( )
%  forest = forestRegrTrain( Xfeat, Xsrc, Xtar, [varargin] )
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
% See also: forestRegrApply, forestAltRegrTrain
% 
% The code is adapted from Piotr's Image&Video Toolbox (Copyright Piotr
% Dollar). 

% get additional parameters and fill in remaining parameters
dfs={ 'M',1,  'minChild',64,  'minCount',128,  'N1',[],  'F1',[],  ...
  'F2',5,  'maxDepth',64,  'fWts',[],  'splitfuntype','pair',  ...
  'nodesubsample',1000,  'splitevaltype','variance', ...
  'lambda',0.01,  'estimatelambda',0,  'kappa',1,  'leaflearntype','linear', ...
  'usepf',0,  'verbose',0 };
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

% train M random trees on different subsets of data
dWtsUni = ones(1,N,'single'); dWtsUni=dWtsUni/sum(dWtsUni);
if opts.usepf
  
  tmp_forests=cell(opts.M,1);
  parfor i=1:opts.M
    if(N==opts.N1), d=1:N; else d=wswor(dWtsUni,opts.N1,4); end %#ok<PFBNS>
    Xfeat1=Xfeat(:,d); Xsrc1=Xsrc(:,d); Xtar1=Xtar(:,d); %#ok<PFBNS>
    tmp_forests{i}=treeRegrTrain(Xfeat1,Xsrc1,Xtar1,opts);
  end
  for i=1:opts.M, if(i==1), tmp=tmp_forests{i}; forest=tmp(ones(opts.M,1));
    else forest(i)=tmp_forests{i}; end; end
  
else
  
  for i=1:opts.M
    if(N==opts.N1), d=1:N; else d=wswor(dWtsUni,opts.N1,4); end
    Xfeat1=Xfeat(:,d); Xsrc1=Xsrc(:,d); Xtar1=Xtar(:,d);
    tree = treeRegrTrain(Xfeat1,Xsrc1,Xtar1,opts);
    if(i==1), forest=tree(ones(opts.M,1)); else forest(i)=tree; end
  end
  
end
end


% =========================================================================
% ========= helper function ===============================================

function tree = treeRegrTrain( Xfeat, Xsrc, Xtar, opts )
% Train a single regression tree

% define some constants and the tree model
[~,N]=size(Xfeat); K=2*N-1;
thrs=zeros(K,1,'single'); 
if strcmp(opts.splitfuntype,'single')
  fids=zeros(K,1,'uint32');
elseif strcmp(opts.splitfuntype,'pair')
  fids=zeros(K,2,'uint32');
else
  error('Unknown splitfunction type');
end
child=zeros(K,1,'uint32'); count=child; depth=child;
leafinfo = struct('T',[],'type',-1,'id',-1); leafinfo = leafinfo(ones(K,1));
dids=cell(K,1); dids{1}=uint32(1:N); k=1; K=2;
% frequence for printing a status message (if in verbose mode)
msgnodestep = 200;

% train the tree
while( k < K )
  % get node data
  dids1=dids{k}; count(k)=length(dids1); XfeatNode=Xfeat(:,dids1);
  XsrcNode = Xsrc(:,dids1); XtarNode = Xtar(:,dids1);
  if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), ...
      fprintf('Node %04d, depth %02d, %07d samples () ',k,depth(k),count(k)); end
  
  % if insufficient data or max-depth reached, don't train split
  if( count(k)<=opts.minCount||depth(k)>opts.maxDepth||count(k)<(2*opts.minChild) )
    if (opts.verbose && (mod(k,msgnodestep)==0)||k==1), ...
        fprintf('becomes a leaf (stop criterion active)\n'); end
    leafinfo(k)=createLeaf(XsrcNode,XtarNode,opts.leaflearntype,opts.lambda,opts.estimatelambda);
    k=k+1; continue;
  end
  if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), fprintf('find split () '); end
  
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
  if opts.nodesubsample > 0 && opts.nodesubsample < count(k)
    randinds = randsample(count(k),opts.nodesubsample);
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
  if ~isinf(rerr) && count0>=opts.minChild && (count(k)-count0)>=opts.minChild
    validsplit=true;
  end
  
  % continue tree training (either split or create a leaf)
  if validsplit
    child(k)=K; fids(k,:)=fid; thrs(k)=thr; dids{K}=dids1(left); 
    dids{K+1}=dids1(~left); depth(K:K+1)=depth(k)+1; K=K+2; 
    dids{k}=[]; % delete the dids as we have a split node here
    if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), ...
        fprintf('valid split (loss=%.6f)\n',rerr); end;
  else
    if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), ...
        fprintf('invalid split -> leaf\n'); end
    leafinfo(k)=createLeaf(XsrcNode,XtarNode,opts.leaflearntype,opts.lambda,opts.estimatelambda);
  end
  k=k+1;
end; K=K-1;

% create output model struct
tree=struct('fids',fids(1:K,:),'thrs',thrs(1:K),'child',child(1:K),...
  'count',count(1:K),'depth',depth(1:K),'leafinfo',leafinfo(1:K),...
  'dids',[]);

% create the leaf-node id mapping
leafcnt = 0;
for i=1:length(tree.leafinfo)
  if ~isempty(tree.leafinfo(i).T), leafcnt=leafcnt+1; ...
      tree.leafinfo(i).id = leafcnt; end;
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
      %  trerrL = sum(var(XtarL,0,2))/Ft; trerrR = sum(var(XtarR,0,2))/Ft;
      %  trerr = (nl*trerrL + nr*trerrR)/(nl+nr);
      %case 'variancefull'
        trerrL = sum(var(XtarL,0,2))/Ft; trerrR = sum(var(XtarR,0,2))/Ft;
        if kappa>0
          trerrLsrc=sum(var(XsrcL,0,2))/Fs; trerrRsrc=sum(var(XsrcR,0,2))/Fs;
          trerrL=(trerrL+kappa*trerrLsrc)/2; trerrR=(trerrR+kappa*trerrRsrc)/2;
        end
        trerr = (nl*trerrL + nr*trerrR)/(nl+nr);
      case 'reconstruction' % based on a sampled dictionary
      %  XsrcL = [XsrcL; ones(1,size(XsrcL,2),'single')]; %#ok<AGROW>
      %  TL =  XtarL * ((XsrcL*XsrcL' + lambda*eye(size(XsrcL,1))) \ XsrcL)';
      %  XsrcR = [XsrcR; ones(1,size(XsrcR,2),'single')]; %#ok<AGROW>
      %  TR =  XtarR * ((XsrcR*XsrcR' + lambda*eye(size(XsrcR,1))) \ XsrcR)';
      %  trerrL = sqrt(sum(sum((XtarL-TL*XsrcL).^2))/nl);
      %  trerrR = sqrt(sum(sum((XtarR-TR*XsrcR).^2))/nr);
      %  trerr = (nl*trerrL + nr*trerrR) / (nl+nr);
      %case 'reconstructionfull'
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
