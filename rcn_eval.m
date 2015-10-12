% evaluation code for rcn

% Initialize code
clear;
p = pwd;
addpath(fullfile(p, 'methods'));  % the upscaling methods
addpath(fullfile(p, 'utils'));  % utils
% addpath(fullfile(p, 'ompbox'));  % Orthogonal Matching Pursuit
% run('../scripts/vlfeat-0.9.20/toolbox/vl_setup');

% check for dataset
dataset = {'Set5'};
numDataset = numel(dataset);
for i = 1:numDataset
    if ~exist(fullfile('data',dataset{i}), 'dir')
        error(['There is no dataset : ', dataset{i}.name]);
    end
end

% Experiment Set
       evalSetting = evalSet('Bicubic', 'Bicubic', 'Set5', 2, []);
evalSetting(end+1) = evalSet('Bicubic', 'Bicubic', 'Set5', 3, []);
evalSetting(end+1) = evalSet('Bicubic', 'Bicubic', 'Set5', 4, []);

evalSetting(end+1) = evalSet('SRCNN', 'SRCNN', 'Set5', 2, []);
evalSetting(end+1) = evalSet('SRCNN', 'SRCNN', 'Set5', 3, []);
evalSetting(end+1) = evalSet('SRCNN', 'SRCNN', 'Set5', 4, []);

evalSetting(end+1) = evalSet('A+', 'A+', 'Set5', 2, []);
evalSetting(end+1) = evalSet('A+', 'A+', 'Set5', 3, []);
evalSetting(end+1) = evalSet('A+', 'A+', 'Set5', 4, []);

evalSetting(end+1) = evalSet('RCN_basic', 'RCN', 'Set5', 2, []);
evalSetting(end+1) = evalSet('RCN_basic', 'RCN', 'Set5', 3, []);
evalSetting(end+1) = evalSet('RCN_basic', 'RCN', 'Set5', 4, []);
                                                           %¦¦-- model path option is not implemented yet.

% Setup outDir
outDir = 'data\result';
if ~exist('data\result', 'dir'), mkdir('data/result'); end

%--------------------------------------------------------------------------
% 1. Make SR images & save them for every eval settings
%--------------------------------------------------------------------------
for indEvalSetting = 1:numel(evalSetting)
    eval_SR(evalSetting(indEvalSetting), outDir);
end

%--------------------------------------------------------------------------
% 2. Compute Quantitive results & draw tables
%--------------------------------------------------------------------------
dataset = {'Set5'};
sf = [3 4];
exp = {'Bicubic', 'SRCNN', 'A+', 'RCN_basic'};

for indDataset = 1:numel(dataset)
    datasetName = dataset{indDataset};
    gtDir = fullfile('data',datasetName);
    outDir = fullfile('data','result');
    img_lst = dir(gtDir); img_lst = img_lst(3:end);
    numImg = numel(img_lst);
    
    for indSF = 1:numel(sf)
        SF = sf(indSF);
        PSNR_table = zeros(numImg, numel(exp));
        SSIM_table = zeros(numImg, numel(exp));
        
        for indImg = 1:numel(img_lst)
            [~,imgName,imgExt] = fileparts(img_lst(indImg).name);
            imGT = imread(fullfile(gtDir, [imgName,imgExt]));

            for indExp = 1:numel(exp)
                expName = exp{indExp};
                outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
                imSR = imread(fullfile(outDir, outRoute, [imgName,'.png']));
                
                [psnr, ssim] = compute_diff(imGT, imSR, SF);
                PSNR_table(indImg, indExp) = psnr;
                SSIM_table(indImg, indExp) = ssim;
            end
        end
        
        avgPSNR = mean(PSNR_table,1);
        avgSSIM = mean(SSIM_table,1);
        
        fprintf('\n\n=== Quantitative results for dataset %s on SRF %d === \n\n', datasetName, SF);
        fprintf('Peak signal-to-noise ratio (PSNR) \n')
        fprintf('      %8s\t%8s\t%8s\t%8s\t\n', exp{1}, exp{2}, exp{3}, exp{4});
        fprintf('PSNR|%8.02f\t|%8.02f\t|%8.02f\t|%8.02f\t| \n', ...
            avgPSNR(1), avgPSNR(2), avgPSNR(3), avgPSNR(4));
    end
end

% ToDo :
% Table 1 : Set5 detail as tex. (PSNR, SSIM, time)
% Table 2 : Set5 Set14 B100 average as tex. (PSNR, SSIM, time)
% Fig 1 : Good Qulitative result (zoomed)





        