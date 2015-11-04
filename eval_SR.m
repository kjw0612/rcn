function eval_SR(evalSetting, outDir)

outRoute = fullfile(evalSetting.expName, [evalSetting.expName,'_',evalSetting.dataset,'_x',num2str(evalSetting.sf)]);
if strcmp(evalSetting.method,'RCNInter')
    outRoute = fullfile(evalSetting.expName, [evalSetting.expName,'_',evalSetting.dataset,'_x',num2str(evalSetting.sf),'_d',num2str(evalSetting.opts)]);
end
outRoute = fullfile(outDir, outRoute);

switch evalSetting.dataset
    case 'Set5'
        effImgNum = 5;
    case 'Set14'
        effImgNum = 14;
    case 'B100'
        effImgNum = 100;
    case 'Urban100'
        effImgNum = 100;
    otherwise
        disp('Unknown dataset name');
end

if exist(outRoute, 'dir')
else
    mkdir(outRoute);
end

if numel(dir(fullfile(outRoute, '*.png'))) == effImgNum
    disp(['This experiment was done already. @: ', outRoute, ' please check if re-evaluation is required']);       
else    
    switch evalSetting.method
        case 'Bicubic'
            Bicubic(evalSetting.dataset, evalSetting.sf, evalSetting.model, outRoute);
        case 'A+'
            Aplus(evalSetting.dataset, evalSetting.sf, evalSetting.model, outRoute);
        case 'SRCNN'
            SRCNN(evalSetting.dataset, evalSetting.sf, evalSetting.model, outRoute);
        case 'RFL'
            RFL(evalSetting.dataset, evalSetting.sf, evalSetting.model, outRoute);
        case 'SelfEx'
            SelfEx(evalSetting.dataset, evalSetting.sf, outRoute);
        case 'RCN'
            RCN_(evalSetting.dataset, evalSetting.sf, evalSetting.model, outRoute);
        case 'RCNInter'
            RCNInter(evalSetting.dataset, evalSetting.sf, evalSetting.model, evalSetting.opts, outRoute);
        otherwise
            disp('Unknown method name');
    end
end
