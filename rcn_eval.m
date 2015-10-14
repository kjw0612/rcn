function rcn_eval()% evaluation code for rcn
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
%        evalSetting = evalSet('Bicubic', 'Bicubic', 'Set5', 2, []);
% evalSetting(end+1) = evalSet('Bicubic', 'Bicubic', 'Set5', 3, []);
% evalSetting(end+1) = evalSet('Bicubic', 'Bicubic', 'Set5', 4, []);
% 
% evalSetting(end+1) = evalSet('SRCNN', 'SRCNN', 'Set5', 2, []);
% evalSetting(end+1) = evalSet('SRCNN', 'SRCNN', 'Set5', 3, []);
% evalSetting(end+1) = evalSet('SRCNN', 'SRCNN', 'Set5', 4, []);
% 
% evalSetting(end+1) = evalSet('A+', 'A+', 'Set5', 2, []);
% evalSetting(end+1) = evalSet('A+', 'A+', 'Set5', 3, []);
% evalSetting(end+1) = evalSet('A+', 'A+', 'Set5', 4, []);
% 
% evalSetting(end+1) = evalSet('RCN basic', 'RCN', 'Set5', 2, []);
% evalSetting(end+1) = evalSet('RCN basic', 'RCN', 'Set5', 3, []);
% evalSetting(end+1) = evalSet('RCN basic', 'RCN', 'Set5', 4, []);
%                                                            %¦¦-- model path option is not implemented yet.
do.dataset = {'Set5','Set14'};
do.sf = [2 3 4];
do.exp = {{'Bicubic', 'Bicubic'},{'SRCNN', 'SRCNN'},{'A+', 'A+'},{'RCN basic', 'RCN'}};
for i = 1:numel(do.dataset)
    for j = 1:numel(do.sf)
        for k = 1:numel(do.exp)
            if ~exist('evalSetting','var')
                evalSetting = evalSet(do.exp{k}{1},do.exp{k}{2},do.dataset{i},do.sf(j),[]);
            else
                evalSetting(end+1) = evalSet(do.exp{k}{1},do.exp{k}{2},do.dataset{i},do.sf(j),[]);
            end
        end
    end
end

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
% 2. Compute Quantitive results & draw tables. Here's table type 1.
%--------------------------------------------------------------------------
dataset = {'Set5'};
sf = [2 3 4];
exp = {'Bicubic', 'SRCNN', 'A+', 'RCN basic'};
tableName = 'table_1';

fid = fopen([tableName,'.tex'],'w');
fprintf(fid,'\\documentclass{article}\n');
fprintf(fid,'\\usepackage[english]{babel}\n');
fprintf(fid,'\\usepackage{multirow}\n');
fprintf(fid,'\\usepackage{color}\n\n');
fprintf(fid,'\\begin{document}\n\n');
fprintf(fid,'\\begin{table}\n\\begin{center}\n');
fprintf(fid,'\\begin{tabular}{ |');
for indColumn = 1:numel(exp)+2
    fprintf(fid,' c |');
end
fprintf(fid,' }\n\\hline\n');

%for indDataset = 1:numel(dataset)
datasetName = dataset{1};
gtDir = fullfile('data',datasetName);
outDir = fullfile('data','result');
img_lst = dir(gtDir); img_lst = img_lst(3:end);
numImg = numel(img_lst);


fprintf(fid,[datasetName, ' & scale']);
for indExp = 1:numel(exp)
    fprintf(fid,[' & ',exp{indExp}]);
end
fprintf(fid,'\\\\\n');    


for indSF = 1:numel(sf)
    fprintf(fid,'\\hline\n');
    SF = sf(indSF);
    PSNR_table = zeros(numImg, numel(exp));
    SSIM_table = zeros(numImg, numel(exp));
    imgNames = cell(numImg,1);

    for indImg = 1:numImg
        [~,imgNames{indImg},imgExt] = fileparts(img_lst(indImg).name);
        imGT = imread(fullfile(gtDir, [imgNames{indImg},imgExt]));            

        for indExp = 1:numel(exp)
            expName = exp{indExp};
            outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
            imSR = imread(fullfile(outDir, outRoute, [imgNames{indImg},'.png']));

            [psnr, ssim] = compute_diff(imGT, imSR, SF);
            PSNR_table(indImg, indExp) = psnr;
            SSIM_table(indImg, indExp) = ssim;
        end
    end

    [~,maxPSNR] = max(PSNR_table,[],2);
    [~,secmaxPSNR] = secmax(PSNR_table,2);
    avgPSNR = mean(PSNR_table,1);
    avgSSIM = mean(SSIM_table,1);

    fprintf('\n\n=== Quantitative results for dataset %s on SRF %d === \n\n', datasetName, SF);
    fprintf('Peak signal-to-noise ratio (PSNR) \n')
    fprintf('      %8s\t%8s\t%8s\t%8s\t\n', exp{1}, exp{2}, exp{3}, exp{4});
    fprintf('PSNR|%8.02f\t|%8.02f\t|%8.02f\t|%8.02f\t| \n', ...
        avgPSNR(1), avgPSNR(2), avgPSNR(3), avgPSNR(4));

    for indImg = 1:numImg
        fprintf(fid, [setValidName(imgNames{indImg},'_'), ' & $\\times$', num2str(SF)]);            
        for indExp = 1:numel(exp)
            if indExp == maxPSNR(indImg)
                fprintf(fid, [' & {\\color{red}',num2str(PSNR_table(indImg, indExp),'%.2f'),'}']);
            elseif indExp == secmaxPSNR(indImg)
                fprintf(fid, [' & {\\color{blue}',num2str(PSNR_table(indImg, indExp),'%.2f'),'}']);
            else
                fprintf(fid, [' & ',num2str(PSNR_table(indImg, indExp),'%.2f')]);
            end
        end
        fprintf(fid, '\\\\\n');
    end
    fprintf(fid,'\\hline\n');
end
%end
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,['\\caption{PSNR for scale factor $\\times$',num2str(SF),' for ',datasetName, ... 
    '. {\\color{red}Red color} indicates the best performance and {\\color{blue}blue color} indicates the second best one.}\n']);
fprintf(fid,'\\end{center}\n\\end{table}\n\n');  
fprintf(fid,'\\end{document}');    
fclose(fid);    


%--------------------------------------------------------------------------
% table type 2.
%--------------------------------------------------------------------------
dataset = {'Set5','Set14'};
sf = [2 3 4];
exp = {'Bicubic', 'SRCNN', 'A+', 'RCN basic'};
tableName = 'table_2';

fid = fopen([tableName,'.tex'],'w');
fprintf(fid,'\\documentclass{article}\n');
fprintf(fid,'\\usepackage[english]{babel}\n');
fprintf(fid,'\\usepackage{multirow}\n');
fprintf(fid,'\\usepackage{color}\n\n');
fprintf(fid,'\\begin{document}\n\n');
fprintf(fid,'\\begin{table}\n\\begin{center}\n');
fprintf(fid,'\\begin{tabular}{ |');
for indColumn = 1:numel(exp)+2
    fprintf(fid,' c |');
end
fprintf(fid,' }\n\\hline\n');
fprintf(fid,[' ', ' & scale']);
for indExp = 1:numel(exp)
    fprintf(fid,[' & ',exp{indExp}]);
end
fprintf(fid,'\\\\\n\\hline\n');

for indDataset = 1:numel(dataset)
    datasetName = dataset{indDataset};
    gtDir = fullfile('data',datasetName);
    outDir = fullfile('data','result');
    img_lst = dir(gtDir); img_lst = img_lst(3:end);
    numImg = numel(img_lst);    

    fprintf(fid, ['\\multirow{',num2str(numel(sf)),'}{*}{',setValidName(datasetName,'_'),'}']);
    for indSF = 1:numel(sf)
        SF = sf(indSF);
        PSNR_table = zeros(numImg, numel(exp));
        SSIM_table = zeros(numImg, numel(exp));
        imgNames = cell(numImg,1);
        
        for indImg = 1:numImg
            [~,imgNames{indImg},imgExt] = fileparts(img_lst(indImg).name);
            imGT = imread(fullfile(gtDir, [imgNames{indImg},imgExt]));            

            for indExp = 1:numel(exp)
                expName = exp{indExp};
                outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
                imSR = imread(fullfile(outDir, outRoute, [imgNames{indImg},'.png']));
                
                [psnr, ssim] = compute_diff(imGT, imSR, SF);
                PSNR_table(indImg, indExp) = psnr;
                SSIM_table(indImg, indExp) = ssim;
            end
        end
        
        avgPSNR = mean(PSNR_table,1);
        avgSSIM = mean(SSIM_table,1);
        [~,maxAvgPSNR] = max(avgPSNR,[],2);
        [~,secmaxAvgPSNR] = secmax(avgPSNR,2);        
                
        fprintf(fid,[' & $\\times$', num2str(SF)]);            
        for indExp = 1:numel(exp)
            if indExp == maxAvgPSNR
                fprintf(fid, [' & {\\color{red}',num2str(avgPSNR(1, indExp),'%.2f'),'}']);
            elseif indExp == secmaxAvgPSNR
                fprintf(fid, [' & {\\color{blue}',num2str(avgPSNR(1, indExp),'%.2f'),'}']);
            else
                fprintf(fid, [' & ',num2str(avgPSNR(1, indExp),'%.2f')]);
            end
        end
        fprintf(fid, '\\\\\n');        
    end
    fprintf(fid,'\\hline\n');
end
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,['\\caption{Average PSNR for scale factor $\\times$',num2str(SF),' for ']);
for i=1:numel(dataset)    
    if i < numel(dataset)-1
        fprintf(fid,[dataset{i}, ', ']);
    elseif i == numel(dataset)-1
        fprintf(fid,[dataset{i}, ' and ']);
    else
        fprintf(fid,dataset{i});
    end
end
fprintf(fid,'. {\\color{red}Red color} indicates the best performance and {\\color{blue}blue color} indicates the second best one.}\n');
fprintf(fid,'\\end{center}\n\\end{table}\n\n');  
fprintf(fid,'\\end{document}');    
fclose(fid);    

% ToDo :
% Fig 1 : Good Qulitative result (zoomed)

function validName = setValidName(name, exp)
ind = regexp(name,exp);
if ind > 0
    validName = name(1:ind-1);
else
    validName = name;
end
    
function [sec,seci] = secmax(A,dim)
maxIndMatrix = bsxfun(@eq,A,max(A,[],2));
A(maxIndMatrix) = -Inf;
[sec,seci] = max(A,[],dim);

        