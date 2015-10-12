function eval_SR(evalSetting, outDir)

outRoute = fullfile(evalSetting.expName, [evalSetting.expName,'_',evalSetting.dataset,'_x',num2str(evalSetting.sf)]);
outRoute = fullfile(outDir, outRoute);

if exist(outRoute, 'dir')
    disp(['Already exist dir name : ', outRoute, ' please check if re-evaluation is required']);   
else
    mkdir(outRoute);
    
    switch evalSetting.method
        case 'Bicubic'
            Bicubic(evalSetting.dataset, evalSetting.sf, evalSetting.model, outRoute);
        case 'A+'
            Aplus(evalSetting.dataset, evalSetting.sf, evalSetting.model, outRoute);
        case 'SRCNN'
            SRCNN(evalSetting.dataset, evalSetting.sf, evalSetting.model, outRoute);
        case 'RCN'
            RCN_(evalSetting.dataset, evalSetting.sf, evalSetting.model, outRoute);
        otherwise
            disp('Unknown method name');
    end
end
