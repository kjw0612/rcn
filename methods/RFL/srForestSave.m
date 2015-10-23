function srForestSave( modfile, srforestStore, verbose )
% srForestSave Saves a model to the given model file path. 
% 
% Given a model and a path for storage, this function saves the model to
% HDD. Use this function (in conjunction with srForestLoad) instead of 
% directly calling Matlab's save and load functions for smaller model files 
% and faster loading. 
% 
% INPUTS
%  modfile        - [REQ] path to the model
%  srforestStore  - [REQ] the model to be stored
%  verbose        - [0] print status messages or not 
% 
% See also: srForestLoad, srForestTrain, srForestApply
% 

if nargin<3, verbose=0; end;

if verbose, fprintf('Saving the model\n'); end;
Tmats = cell(length(srforestStore.model),3);
for t=1:length(srforestStore.model)
  if verbose
    fprintf('  preparing tree %02d/%02d\n',t,length(srforestStore.model));
  end
  tree=srforestStore.model(t);
  Tmats{t,1} = {tree.leafinfo(:).T}';
  Tmats{t,2} = cat(1,tree.leafinfo(:).type);
  Tmats{t,3} = cat(1,tree.leafinfo(:).id);
  tree.leafinfo = [];
  srforestStore.model(t)=tree;
end
if verbose, fprintf('Saving the model to HDD ... \n'); end;
save(modfile,'srforestStore','Tmats','-v7.3');
