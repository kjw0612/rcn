%--------------------------------------------------------------------------      
function evalSetting = evalSet(expName, method, dataset, sf, model)
%--------------------------------------------------------------------------'
    evalSetting = struct();
    evalSetting.expName = expName;
    evalSetting.method = method;
    evalSetting.dataset = dataset;
    evalSetting.sf = sf;
    evalSetting.model = model;