%--------------------------------------------------------------------------      
function evalSetting = evalSet(expName, method, dataset, sf, model, opts)
%--------------------------------------------------------------------------'
switch nargin
    case 4
        model = [];
        opts = [];        
    case 5
        opts = [];        
end
evalSetting = struct();
evalSetting.expName = expName;
evalSetting.method = method;
evalSetting.dataset = dataset;
evalSetting.sf = sf;
evalSetting.model = model;
evalSetting.opts = opts;
