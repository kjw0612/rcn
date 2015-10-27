function rcn_eval()% evaluation code for rcn
% Initialize code
clear;
p = pwd;
addpath(genpath(fullfile(p, 'methods')));  % the upscaling methods
addpath(fullfile(p, 'utils'));  % utils
addpath(genpath(fullfile(p, 'toolbox')));
run('snudeep/matlab/vl_setupnn.m');
% addpath(fullfile(p, 'ompbox'));  % Orthogonal Matching Pursuit
% run('../scripts/vlfeat-0.9.20/toolbox/vl_setup');

% check for dataset
dataset = {'Set5','Set14'};
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
% evalSetting(end+1) = evalSet('RCN basic', 'RCN', 'Set5', 2, []);
% evalSetting(end+1) = evalSet('RCN basic', 'RCN', 'Set5', 3, []);
% evalSetting(end+1) = evalSet('RCN basic', 'RCN', 'Set5', 4, []);
                                                             %��-- model path option is not implemented yet.
do.dataset = {'Set5','Set14','B100','Urban100'};
do.sf = [2 3 4];
do.exp = {{'Bicubic', 'Bicubic'},{'SRCNN', 'SRCNN'},{'A+', 'A+'},{'RFL', 'RFL'}};
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
% evalSetting(end+1) = evalSet('RCN 256', 'RCN', 'Set5', 3, 'best256.mat');
% evalSetting(end+1) = evalSet('RCN 256', 'RCN', 'Set14', 3, 'best256.mat');
evalSetting(end+1) = evalSet('RCN 64', 'RCN', 'Set5', 3, 'best64.mat');
evalSetting(end+1) = evalSet('RCN 64', 'RCN', 'Set14', 3, 'best64.mat');
evalSetting(end+1) = evalSet('RCN 64', 'RCN', 'B100', 3, 'best64.mat');
evalSetting(end+1) = evalSet('RCN 64', 'RCN', 'Urban100', 3, 'best64.mat');
% Setup outDir
outDir = 'data/result';
if ~exist('data/result', 'dir'), mkdir('data/result'); end

fileID = fopen('rcn_eval_test.tex','w');
%--------------------------------------------------------------------------
% 1. Make SR images & save them for every eval settings
%--------------------------------------------------------------------------
for indEvalSetting = 1:numel(evalSetting)
    eval_SR(evalSetting(indEvalSetting), outDir);
end

%--------------------------------------------------------------------------
% 2. Compute Quantitive results & draw tables. Here's table type 1.
%--------------------------------------------------------------------------
t1opts.dataset = 'Set5';
t1opts.problem = 'SR';
t1opts.sf = [3];
t1opts.exp = {'Bicubic', 'A+','SRCNN', 'RFL', 'SelfEx', 'RCN 256'};
t1opts.printTime = true;
t1opts.tableName = 'table_1';
t1opts.fid = fileID;%fopen([t1opts.tableName,'.tex'],'w');

%--------------------------------------------------------------------------
% table type 2.
%--------------------------------------------------------------------------
t2opts.dataset = {'Set5','Set14','B100'};
t2opts.problem = 'SR';
t2opts.sf = [3];
t2opts.exp = {'Bicubic', 'A+','SRCNN', 'RFL', 'SelfEx', 'RCN 64'};
t2opts.printTime = true;
t2opts.tableName = 'table_2';
t2opts.fid = fileID;

%--------------------------------------------------------------------------
% figure type 1. 
%
% t------tt------tt------tt------t
% | HR   |t------tt------tt------t
% |      |t------tt------tt------t
% t------tt------tt------tt------t
%--------------------------------------------------------------------------
f1opts.dataset = 'Set5';
f1opts.imgNum = 1;
f1opts.boxSize = [60 100];
f1opts.boxPose = [];
f1opts.lineWidth = 2;
f1opts.lineColor = [255 0 0];
f1opts.problem = 'SR';
f1opts.sf = 3;
f1opts.exp = {'HR','Bicubic','Bicubic','A+','SRCNN','RCN 256'};
f1opts.figName = 'fig1';
f1opts.figDir = 'paper/figs';
f1opts.fid = fileID;

%--------------------------------------------------------------------------
% figure type 2. 
%
% ��------����------����------����------����------��
% ��      ����      ����      ����      ����      ��
% ��------����------����------����------����------��
% ��      ����      ����      ����      ����      ��
% ��------����------����------����------����------��
%--------------------------------------------------------------------------
f2opts.dataset = 'Set5';
f2opts.imgNum = 1;
f2opts.boxSize = [60 60];
f2opts.boxPose = [];%[200 150];
f2opts.lineWidth = 2;
f2opts.lineColor = [255 0 0];
f2opts.problem = 'SR';
f2opts.sf = 3;
f2opts.exp = {'HR','Bicubic','A+','SRCNN','RCN 256'};
f2opts.figName = 'fig2';
f2opts.figDir = 'paper/figs';
f2opts.fid = fileID;

%--------------------------------------------------------------------------

texPrefix(fileID);
makeTable1(t1opts);
makeTable2(t2opts);
makeFigure1(f1opts);
makeFigure2(f2opts);
texSuffix(fileID);
fclose(fileID);

%--------------------------------------------------------------------------

function makeTable1(opts)

dataset = opts.dataset;
problem = opts.problem;
sf = opts.sf;
exp = opts.exp;
tableName = opts.tableName;
fid = opts.fid;
printTime = opts.printTime;

%fid = fopen([tableName,'.tex'],'w');
fprintf(fid,'\\begin{table}\n\\begin{center}\n');
fprintf(fid,'\\setlength{\\tabcolsep}{2pt}\n');
if numel(exp) >= 5
    fprintf(fid,'\\scriptsize\n');
else
    fprintf(fid,'\\small\n');
end
fprintf(fid,'\\begin{tabular}{ |');
for indColumn = 1:numel(exp)+2
    fprintf(fid,' c |');
end
fprintf(fid,' }\n\\hline\n');

%for indDataset = 1:numel(dataset)
datasetName = dataset;
gtDir = fullfile('data',datasetName);
outDir = fullfile('data','result');
img_lst = dir(gtDir); img_lst = img_lst(3:end);
numImg = numel(img_lst);

fprintf(fid,['\\multirow{2}{*}{',datasetName,'} & \\multirow{2}{*}{scale}']);
for indExp = 1:numel(exp)
    fprintf(fid,[' & ',exp{indExp}]);
end
fprintf(fid,'\\\\\n &');
for indExp = 1:numel(exp)
    if printTime
        fprintf(fid, ' & PSNR/SSIM/time');
    else
        fprintf(fid, ' & PSNR/SSIM');
    end
end
fprintf(fid,'\\\\\n\\hline\n');    


for indSF = 1:numel(sf)
    fprintf(fid,'\\hline\n');
    SF = sf(indSF);
    PSNR_table = zeros(numImg, numel(exp));
    SSIM_table = zeros(numImg, numel(exp));
    TIME_table = zeros(numImg, numel(exp));
    imgNames = cell(numImg,1);

    for indImg = 1:numImg
        [~,imgNames{indImg},imgExt] = fileparts(img_lst(indImg).name);
        imGT = imread(fullfile(gtDir, [imgNames{indImg},imgExt]));            

        for indExp = 1:numel(exp)
            expName = exp{indExp};
            outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
            imSR = imread(fullfile(outDir, outRoute, [imgNames{indImg},'.png']));
            elapsedTime = load(fullfile(outDir, outRoute, 'elapsed_time.mat'));

            [psnr, ssim] = compute_diff(imGT, imSR, SF);
            PSNR_table(indImg, indExp) = psnr;
            SSIM_table(indImg, indExp) = ssim;
            TIME_table(indImg, indExp) = elapsedTime.timetable(indImg+2);
        end
    end

    [~,maxPSNR] = max(PSNR_table,[],2); [~,maxSSIM] = max(SSIM_table,[],2);
    [~,secmaxPSNR] = secmax(PSNR_table,2); [~,secmaxSSIM] = secmax(SSIM_table,2);
    avgPSNR = mean(PSNR_table,1);
    avgSSIM = mean(SSIM_table,1);
    avgTIME = mean(TIME_table,1);
    [~,maxAvgPSNR] = max(avgPSNR,[],2); [~,maxAvgSSIM] = max(avgSSIM,[],2);
    [~,secmaxAvgPSNR] = secmax(avgPSNR,2); [~,secmaxAvgSSIM] = secmax(avgSSIM,2);

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
                if indExp == maxSSIM(indImg)
                    fprintf(fid, ['/{\\color{red}',num2str(SSIM_table(indImg, indExp),'%.4f'),'}']);
                elseif indExp == secmaxSSIM(indImg)
                    fprintf(fid, ['/{\\color{blue}',num2str(SSIM_table(indImg, indExp),'%.4f'),'}']);
                else
                    fprintf(fid, ['/',num2str(SSIM_table(indImg, indExp),'%.4f')]);
                end
                if printTime 
                    fprintf(fid, ['/',num2str(TIME_table(indImg, indExp),'%.2f')]);
                end
            elseif indExp == secmaxPSNR(indImg)
                fprintf(fid, [' & {\\color{blue}',num2str(PSNR_table(indImg, indExp),'%.2f'),'}']);
                if indExp == maxSSIM(indImg)
                    fprintf(fid, ['/{\\color{red}',num2str(SSIM_table(indImg, indExp),'%.4f'),'}']);
                elseif indExp == secmaxSSIM(indImg)
                    fprintf(fid, ['/{\\color{blue}',num2str(SSIM_table(indImg, indExp),'%.4f'),'}']);
                else
                    fprintf(fid, ['/',num2str(SSIM_table(indImg, indExp),'%.4f')]);
                end
                if printTime
                    fprintf(fid, ['/',num2str(TIME_table(indImg, indExp),'%.2f')]);
                end
            else
                fprintf(fid, [' & ',num2str(PSNR_table(indImg, indExp),'%.2f')]);
                if indExp == maxSSIM(indImg)
                    fprintf(fid, ['/{\\color{red}',num2str(SSIM_table(indImg, indExp),'%.4f'),'}']);
                elseif indExp == secmaxSSIM(indImg)
                    fprintf(fid, ['/{\\color{blue}',num2str(SSIM_table(indImg, indExp),'%.4f'),'}']);
                else
                    fprintf(fid, ['/',num2str(SSIM_table(indImg, indExp),'%.4f')]);
                end
                if printTime
                    fprintf(fid, ['/',num2str(TIME_table(indImg, indExp),'%.2f')]);
                end
            end
        end
        fprintf(fid, '\\\\\n');
    end
    fprintf(fid,'\\hline\\hline\n');
    fprintf(fid,['average & $\\times$',num2str(SF)]);
    for indExp = 1:numel(exp)
        if indExp == maxAvgPSNR
            fprintf(fid, [' & {\\color{red}',num2str(avgPSNR(1, indExp),'%.2f'),'}']);
            if indExp == maxAvgSSIM
                fprintf(fid, ['/{\\color{red}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
            elseif indExp == secmaxAvgSSIM
                fprintf(fid, ['/{\\color{blue}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
            else
                fprintf(fid, ['/',num2str(avgSSIM(1, indExp),'%.4f')]);
            end
            if printTime
                fprintf(fid, ['/',num2str(avgTIME(1, indExp),'%.2f')]);
            end
        elseif indExp == secmaxAvgPSNR
            fprintf(fid, [' & {\\color{blue}',num2str(avgPSNR(1, indExp),'%.2f'),'}']);
            if indExp == maxAvgSSIM
                fprintf(fid, ['/{\\color{red}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
            elseif indExp == secmaxAvgSSIM
                fprintf(fid, ['/{\\color{blue}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
            else
                fprintf(fid, ['/',num2str(avgSSIM(1, indExp),'%.4f')]);
            end
            if printTime
                fprintf(fid, ['/',num2str(avgTIME(1, indExp),'%.2f')]);
            end
        else
            fprintf(fid, [' & ',num2str(avgPSNR(1, indExp),'%.2f')]);
            if indExp == maxAvgSSIM
                fprintf(fid, ['/{\\color{red}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
            elseif indExp == secmaxAvgSSIM
                fprintf(fid, ['/{\\color{blue}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
            else
                fprintf(fid, ['/',num2str(avgSSIM(1, indExp),'%.4f')]);
            end
            if printTime
                fprintf(fid, ['/',num2str(avgTIME(1, indExp),'%.2f')]);
            end
        end
    end
    fprintf(fid,'\\\\\n');
    fprintf(fid,'\\hline\n');
end
%end
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,['\\caption{PSNR for scale factor $\\times$',num2str(SF),' for ',datasetName, ... 
    '. {\\color{red}Red color} indicates the best performance and {\\color{blue}blue color} indicates the second best one.}\n']);
fprintf(fid,'\\end{center}\n\\end{table}\n\n');  

function makeTable2(opts)

dataset = opts.dataset;
problem = opts.problem;
sf = opts.sf;
exp = opts.exp;
tableName = opts.tableName;
fid = opts.fid;
printTime = opts.printTime;

fprintf(fid,'\\begin{table}\n\\begin{center}\n');
fprintf(fid,'\\setlength{\\tabcolsep}{2pt}\n');
if numel(exp) >= 5
    fprintf(fid,'\\scriptsize\n');
else
    fprintf(fid,'\\small\n');
end
fprintf(fid,'\\begin{tabular}{ |');
for indColumn = 1:numel(exp)+2
    fprintf(fid,' c |');
end
fprintf(fid,' }\n\\hline\n');
fprintf(fid,'\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{scale}');
for indExp = 1:numel(exp)
    fprintf(fid,[' & ',exp{indExp}]);
end
fprintf(fid,'\\\\\n &');
for indExp = 1:numel(exp)
    if printTime
        fprintf(fid, ' & PSNR/SSIM/time');
    else
        fprintf(fid, ' & PSNR/SSIM');
    end
end
fprintf(fid,'\\\\\n\\hline\n');

for indDataset = 1:numel(dataset)
    fprintf(fid,'\\hline\n');
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
        TIME_table = zeros(numImg, numel(exp));
        imgNames = cell(numImg,1);
        
        for indImg = 1:numImg
            [~,imgNames{indImg},imgExt] = fileparts(img_lst(indImg).name);
            imGT = imread(fullfile(gtDir, [imgNames{indImg},imgExt]));            

            for indExp = 1:numel(exp)
                expName = exp{indExp};
                outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
                imSR = imread(fullfile(outDir, outRoute, [imgNames{indImg},'.png']));
                elapsedTime = load(fullfile(outDir, outRoute, 'elapsed_time.mat'));

                [psnr, ssim] = compute_diff(imGT, imSR, SF);
                PSNR_table(indImg, indExp) = psnr;
                SSIM_table(indImg, indExp) = ssim;
                TIME_table(indImg, indExp) = elapsedTime.timetable(indImg+2);
            end
        end
        
        avgPSNR = mean(PSNR_table,1);
        avgSSIM = mean(SSIM_table,1);
        avgTIME = mean(TIME_table,1);
        [~,maxAvgPSNR] = max(avgPSNR,[],2); [~,maxAvgSSIM] = max(avgSSIM,[],2);
        [~,secmaxAvgPSNR] = secmax(avgPSNR,2); [~,secmaxAvgSSIM] = secmax(avgSSIM,2); 
                
        fprintf(fid,[' & $\\times$', num2str(SF)]);            
        for indExp = 1:numel(exp)
            if indExp == maxAvgPSNR
                fprintf(fid, [' & {\\color{red}',num2str(avgPSNR(1, indExp),'%.2f'),'}']);
                if indExp == maxAvgSSIM
                    fprintf(fid, ['/{\\color{red}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
                elseif indExp == secmaxAvgSSIM
                    fprintf(fid, ['/{\\color{blue}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
                else
                    fprintf(fid, ['/',num2str(avgSSIM(1, indExp),'%.4f')]);
                end
                if printTime
                    fprintf(fid, ['/',num2str(avgTIME(1, indExp),'%.2f')]);
                end
            elseif indExp == secmaxAvgPSNR
                fprintf(fid, [' & {\\color{blue}',num2str(avgPSNR(1, indExp),'%.2f'),'}']);
                if indExp == maxAvgSSIM
                    fprintf(fid, ['/{\\color{red}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
                elseif indExp == secmaxAvgSSIM
                    fprintf(fid, ['/{\\color{blue}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
                else
                    fprintf(fid, ['/',num2str(avgSSIM(1, indExp),'%.4f')]);
                end
                if printTime
                    fprintf(fid, ['/',num2str(avgTIME(1, indExp),'%.2f')]);
                end
            else
                fprintf(fid, [' & ',num2str(avgPSNR(1, indExp),'%.2f')]);
                if indExp == maxAvgSSIM
                    fprintf(fid, ['/{\\color{red}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
                elseif indExp == secmaxAvgSSIM
                    fprintf(fid, ['/{\\color{blue}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
                else
                    fprintf(fid, ['/',num2str(avgSSIM(1, indExp),'%.4f')]);
                end
                if printTime
                    fprintf(fid, ['/',num2str(avgTIME(1, indExp),'%.2f')]);
                end
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

function makeFigure1(opts)

dataset = opts.dataset;
imgNum = opts.imgNum;
boxSize = opts.boxSize;
boxPose = opts.boxPose;
lineWidth = opts.lineWidth;
lineColor = opts.lineColor;
problem = opts.problem;
sf = opts.sf;
exp = opts.exp;
figName = opts.figName;
figDir = opts.figDir;
fid = opts.fid;

numColumn = numel(exp)/2;

if ~exist(fullfile(figDir,figName),'dir')
    mkdir(fullfile(figDir,figName));
end
datasetName = dataset;
SF = sf;
gtDir = fullfile('data',datasetName);
outDir = fullfile('data','result');
img_lst = dir(gtDir); img_lst = img_lst(3:end);
[~,imgName,imgExt] = fileparts(img_lst(imgNum).name);
imGT = imread(fullfile(gtDir, [imgName,imgExt]));
imGT = modcrop(imGT, SF);

if isempty(boxPose)
    for indExp = 1:numel(exp)
        expName = exp{indExp};
        outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
        if strcmp(expName,'SRCNN')
            imSRCNN = imread(fullfile(outDir, outRoute, [imgName,'.png']));
        elseif strcmp(expName,'A+')
            imAplus = imread(fullfile(outDir, outRoute, [imgName,'.png']));
        elseif indExp == numel(exp)
            imRCN = imread(fullfile(outDir, outRoute, [imgName,'.png']));
        end
    end
    [boxPose(1),boxPose(2)] = findBestPos(imGT, imSRCNN, imAplus, imRCN, boxSize);
end

PSNR_array = zeros(numel(exp),1);
SSIM_array = zeros(numel(exp),1);
ShapeInserter = vision.ShapeInserter('LineWidth',lineWidth,'BorderColor','Custom','CustomBorderColor',lineColor);

for indExp = 1:numel(exp)
    expName = exp{indExp};
    outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
    if ~strcmp(expName,'HR')
        imSR = imread(fullfile(outDir, outRoute, [imgName,'.png']));
        [psnr, ssim] = compute_diff(imGT, imSR, SF);
        PSNR_array(indExp, 1) = psnr;
        SSIM_array(indExp, 1) = ssim;
        imSRcolor = colorize(imGT, imSR, SF);
    else
        imSRcolor = modcrop(imGT, SF);
    end        
        subimSRcolor = imSRcolor(boxPose(1):boxPose(1)+boxSize(1)-1,boxPose(2):boxPose(2)+boxSize(2)-1,:);
    imwrite(subimSRcolor,fullfile(figDir,figName,[imgName,'_for_',figName,'_',expName,'.png']));
end
imGTbox = step(ShapeInserter, imGT, int32(cat(2,fliplr(boxPose),fliplr(boxSize))));
imwrite(imGTbox,fullfile(figDir,figName,[imgName,'_GTbox','.png']));

fprintf(fid,'\\begin{figure}\n');
fprintf(fid,'\\begin{adjustwidth}{-1cm}{-1cm}\n');
fprintf(fid,'\\begin{center}\n');
fprintf(fid,'\\small\n');
fprintf(fid,'\\setlength{\\tabcolsep}{-1pt}\n');
fprintf(fid,'\\begin{tabular}{ c');
for indColumn = 1:numColumn
    fprintf(fid,' C{3.7cm} ');
end
fprintf(fid,' }\n');
fprintf(fid, ['\\multirow{4}{*}{\\graphicspath{{',figDir,'/',figName,'/}}\\includegraphics[width=0.30\\textwidth]{', ...
             [imgName,'_GTbox','.png'],'}}\n']);
indExp = 0;
indExp2 = 0;
for indRow = 1:4
    for indColumn = 1:numColumn
        if mod(indRow,2) == 1
            indExp = indExp + 1;
            fprintf(fid, ['& \\raisebox{-',num2str(boxSize(1)/7+1,'%.1f'),'ex} {\\graphicspath{{',figDir,'/',figName,'/}}\\includegraphics[width=0.22\\textwidth]{', ...
                         [imgName,'_for_',figName,'_',exp{indExp},'.png'],'}}\\vspace{0.3ex}\n']);            
        else
            indExp2 = indExp2 + 1;
            if strcmp(exp{indExp2},'HR')
                fprintf(fid, ['& ',exp{indExp2},' (PSNR, SSIM)']);
            else
                fprintf(fid, ['& ',exp{indExp2},' (',num2str(PSNR_array(indExp2),'%.2f'),', ',num2str(SSIM_array(indExp2),'%.4f'),')']);
            end
        end
    end    
    fprintf(fid, '\\\\\n');
end
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,'\\end{center}\n');
fprintf(fid,'\\end{adjustwidth}\n');
fprintf(fid,'\\end{figure}\n');

function makeFigure2(opts)

dataset = opts.dataset;
imgNum = opts.imgNum;
boxSize = opts.boxSize;
boxPose = opts.boxPose;
lineWidth = opts.lineWidth;
lineColor = opts.lineColor;
problem = opts.problem;
sf = opts.sf;
exp = opts.exp;
figName = opts.figName;
figDir = opts.figDir;
fid = opts.fid;



numColumn = numel(exp);

if ~exist(fullfile(figDir,figName),'dir')
    mkdir(fullfile(figDir,figName));
end
datasetName = dataset;
SF = sf;
gtDir = fullfile('data',datasetName);
outDir = fullfile('data','result');
img_lst = dir(gtDir); img_lst = img_lst(3:end);
[~,imgName,imgExt] = fileparts(img_lst(imgNum).name);
imGT = imread(fullfile(gtDir, [imgName,imgExt]));
imGT = modcrop(imGT, SF);

if isempty(boxPose)
    for indExp = 1:numel(exp)
        expName = exp{indExp};
        outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
        if strcmp(expName,'SRCNN')
            imSRCNN = imread(fullfile(outDir, outRoute, [imgName,'.png']));
        elseif strcmp(expName,'A+')
            imAplus = imread(fullfile(outDir, outRoute, [imgName,'.png']));
        elseif indExp == numel(exp)
            imRCN = imread(fullfile(outDir, outRoute, [imgName,'.png']));
        end
    end
    [boxPose(1),boxPose(2)] = findBestPos(imGT, imSRCNN, imAplus, imRCN, boxSize);
end

PSNR_array = zeros(numel(exp),1);
SSIM_array = zeros(numel(exp),1);

for indExp = 1:numel(exp)
    expName = exp{indExp};
    outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
    if ~strcmp(expName,'HR')
        imSR = imread(fullfile(outDir, outRoute, [imgName,'.png']));
        [psnr, ssim] = compute_diff(imGT, imSR, SF);
        PSNR_array(indExp, 1) = psnr;
        SSIM_array(indExp, 1) = ssim;
        imSRcolor = colorize(imGT, imSR, SF);
    else
        imSRcolor = modcrop(imGT, SF);
    end
    imSRcolor = shave(imSRcolor,[SF SF]); % for methods like A+, not predicting boundary.
    ShapeInserter = vision.ShapeInserter('LineWidth',lineWidth,'BorderColor','Custom','CustomBorderColor',lineColor);
    boximSRcolor = step(ShapeInserter, imSRcolor, int32(cat(2,fliplr(boxPose),fliplr(boxSize))));
    subimSRcolor = imSRcolor(boxPose(1):boxPose(1)+boxSize(1)-1,boxPose(2):boxPose(2)+boxSize(2)-1,:);
    subimSRcolor = imresize(subimSRcolor, [size(boximSRcolor,1),size(boximSRcolor,2)]);
    ShapeInserter = vision.ShapeInserter('LineWidth',lineWidth*SF,'BorderColor','Custom','CustomBorderColor',lineColor);
    subimSRcolor = step(ShapeInserter, subimSRcolor, int32([1,1,size(subimSRcolor,1),size(subimSRcolor,2)]));
    catimSRcolor = cat(1,boximSRcolor,subimSRcolor);
    imwrite(catimSRcolor,fullfile(figDir,figName,[imgName,'_for_',figName,'_',expName,'.png']));
end

fprintf(fid,'\\begin{figure}\n');
fprintf(fid,'\\begin{adjustwidth}{-1cm}{-1cm}\n');
fprintf(fid,'\\begin{center}\n');
fprintf(fid,'\\small\n');
fprintf(fid,'\\setlength{\\tabcolsep}{3pt}\n');
fprintf(fid,'\\begin{tabular}{ ');
for indColumn = 1:numColumn
    fprintf(fid,' c ');
end
fprintf(fid,' }\n');
for indColumn = 1:numColumn
    if indColumn == 1
        fprintf(fid, ['{\\graphicspath{{',figDir,'/',figName,'/}}\\includegraphics[width=',num2str(1.02/numel(exp),'%.2f'),'\\textwidth]{', ...
                     [imgName,'_for_',figName,'_',exp{indColumn},'.png'],'}}\\vspace{0.3ex}\n']);            
    else
        fprintf(fid, ['& {\\graphicspath{{',figDir,'/',figName,'/}}\\includegraphics[width=',num2str(1.02/numel(exp),'%.2f'),'\\textwidth]{', ...
                     [imgName,'_for_',figName,'_',exp{indColumn},'.png'],'}}\\vspace{0.3ex}\n']);            
    end
   
end
fprintf(fid, '\\\\\n');
for indColumn = 1:numColumn
    if strcmp(exp{indColumn},'HR')
        fprintf(fid, exp{indColumn});
    else
        fprintf(fid, ['& ',exp{indColumn}]);
    end
end
fprintf(fid, '\\\\\n');
for indColumn = 1:numColumn
    if strcmp(exp{indColumn},'HR')
        fprintf(fid, '(PSNR, SSIM)');
    else
        fprintf(fid, ['& (',num2str(PSNR_array(indColumn),'%.2f'),', ',num2str(SSIM_array(indColumn),'%.4f'),')']);
    end
end
fprintf(fid, '\\\\\n');
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,'\\end{center}\n');
fprintf(fid,'\\end{adjustwidth}\n');
fprintf(fid,'\\end{figure}\n');

function texPrefix(fid)

fprintf(fid,'\\documentclass{article}\n');
fprintf(fid,'\\usepackage[english]{babel}\n');
fprintf(fid,'\\usepackage{multirow}\n');
fprintf(fid,'\\usepackage{color}\n');
fprintf(fid,'\\usepackage{graphicx}\n');
fprintf(fid,'\\usepackage[space]{grffile}\n');
fprintf(fid,'\\usepackage{array}\n');
fprintf(fid,'\\newcolumntype{C}[1]{>{\\centering\\arraybackslash}p{#1}}\n');
fprintf(fid,'\\usepackage{chngpage}\n\n');
fprintf(fid,'\\begin{document}\n\n');

function texSuffix(fid)

fprintf(fid,'\\end{document}');


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

function imSRcolor = colorize(imGT, imSR, SF)
if size(imGT,3) < 1
    imSRcolor = imSR;
elseif size(imSR,3) == 3
    imSRcolor = imSR;
else
    imGT = rgb2ycbcr(imGT);
    imGT = modcrop(imGT, SF);
    imSRcolor = cat(3,imSR(:,:,1),imGT(:,:,2),imGT(:,:,3));
    imSRcolor = ycbcr2rgb(imSRcolor);
end

function [maxX, maxY] = findBestPos(imGT, imSRCNN, imAplus, imRCN, boxSize)
% find the region that illustrates where SFFSR works well.
% i.e. maximize PSNR(SFFSR Window ) - max (PSNR(A+ Window),
% PSNR(SRCNNwindow))

if size(imGT,3) > 1
    imGT = rgb2ycbcr(imGT);
    imGT = imGT(:,:,1);
end

max_val = -1e5;
maxX = 1;
maxY = 1;
sz = size(imGT);
stride = 1;

[~, ssim1] = ssim(imRCN,   imGT, 'Exponents', [0 0 1]);
[~, ssim2] = ssim(imSRCNN, imGT, 'Exponents', [0 0 1]);
[~, ssim3] = ssim(imAplus, imGT, 'Exponents', [0 0 1]);
ssim1c = cumsum(cumsum(ssim1, 1), 2);
ssim2c = cumsum(cumsum(ssim2, 1), 2);
ssim3c = cumsum(cumsum(ssim3, 1), 2);
for x = 2:stride:sz(1)-boxSize(1)
    for y=2:stride:sz(2)-boxSize(2)
        val1 = ssim1c(x+boxSize(1)-1,y+boxSize(2)-1) - ssim1c(x+boxSize(1)-1,y-1) - ssim1c(x-1,y+boxSize(2)-1) + 2*ssim1c(x-1,y-1);
        val2 = ssim2c(x+boxSize(1)-1,y+boxSize(2)-1) - ssim2c(x+boxSize(1)-1,y-1) - ssim2c(x-1,y+boxSize(2)-1) + 2*ssim2c(x-1,y-1);
        val3 = ssim3c(x+boxSize(1)-1,y+boxSize(2)-1) - ssim3c(x+boxSize(1)-1,y-1) - ssim3c(x-1,y+boxSize(2)-1) + 2*ssim3c(x-1,y-1);
        if log(val1)-log(max(val2,val3)) > max_val
            max_val= log(val1)-log(max(val2,val3));
            maxX = x;
            maxY = y;
        end
    end
end
        