function srforestStore = srForestLoad( modfile, verbose )
% srForestLoad Loads a model from the given model file path. 
% 
% Given a path to a model file, this function loads the model from HDD. Use
% this function (in conjunction with srForestSave) instead of directly 
% calling Matlab's save and load functions for smaller model files and
% faster loading. 
% 
% INPUTS
%  modfile        - [REQ] path to the model
%  verbose        - [0] print status messages or not 
% 
% OUTPUTS
%  srforest       - the model
% 
% See also: srForestSave, srForestTrain, srForestApply
% 

if nargin<2, verbose=0; end;

if verbose, fprintf('Loading the model file ... \n'); end;
load(modfile,'srforestStore','Tmats');
for t=1:length(srforestStore.model) %#ok<NODEF>
  if verbose
    fprintf(' - preparing tree %d/%d\n',t,length(srforestStore.model));
  end
  tree=srforestStore.model(t);
  leafinfo = struct('T',[],'type',-1,'id',-1); nleafs=size(Tmats{t,2},1); %#ok<USENS>
  leafinfo=leafinfo(ones(nleafs,1));
  % assign the matrix T ...
  tmpCellArray = Tmats{t,1}; [leafinfo.T] = tmpCellArray{:};
  % assign the type
  tmpCellArray = num2cell(Tmats{t,2}); [leafinfo.type] = tmpCellArray{:};
  % assign the id
  tmpCellArray = num2cell(Tmats{t,3}); [leafinfo.id] = tmpCellArray{:};
  tree.leafinfo=leafinfo;
  srforestStore.model(t) = tree;
end
