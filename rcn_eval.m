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
problem = 'SR';
sf = [2];
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
    [~,maxAvgPSNR] = max(avgPSNR,[],2);
    [~,secmaxAvgPSNR] = secmax(avgPSNR,2);        

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
    fprintf(fid,['average & $\\times$',num2str(SF)]);
    for indExp = 1:numel(exp)
        if indExp == maxAvgPSNR
            fprintf(fid, [' & {\\color{red}',num2str(avgPSNR(1, indExp),'%.2f'),'}']);
        elseif indExp == secmaxAvgPSNR
            fprintf(fid, [' & {\\color{blue}',num2str(avgPSNR(1, indExp),'%.2f'),'}']);
        else
            fprintf(fid, [' & ',num2str(avgPSNR(1, indExp),'%.2f')]);
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
fprintf(fid,'\\end{document}');    
fclose(fid);    


%--------------------------------------------------------------------------
% table type 2.
%--------------------------------------------------------------------------
dataset = {'Set5','Set14'};
problem = 'SR';
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
fprintf(fid,['Dataset', ' & scale']);
for indExp = 1:numel(exp)
    fprintf(fid,[' & ',exp{indExp}]);
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

%--------------------------------------------------------------------------
% figure type 1. 
%
% ¦£------¦¤¦£------¦¤¦£------¦¤¦£------¦¤
% ¦¢ HR   ¦¢¦¦------¦¥¦¦------¦¥¦¦------¦¥
% ¦¢      ¦¢¦£------¦¤¦£------¦¤¦£------¦¤
% ¦¦------¦¥¦¦------¦¥¦¦------¦¥¦¦------¦¥
%--------------------------------------------------------------------------
dataset = {'Set5'};
imgNum = 1;
boxSize = [60 100];
boxPose = [200 150];
lineWidth = 2;
lineColor = [255 0 0];
problem = 'SR';
sf = 3;
exp = {'HR','Bicubic','Bicubic','A+','SRCNN','RCN basic'};
figName = 'fig1';
figDir = 'figs';

numColumn = numel(exp)/2;

if ~exist(fullfile(figDir,figName),'dir')
    mkdir(fullfile(figDir,figName));
end
datasetName = dataset{1};
SF = sf;
gtDir = fullfile('data',datasetName);
outDir = fullfile('data','result');
img_lst = dir(gtDir); img_lst = img_lst(3:end);
[~,imgName,imgExt] = fileparts(img_lst(imgNum).name);
imGT = imread(fullfile(gtDir, [imgName,imgExt]));
PSNR_array = zeros(numel(exp),1);
SSIM_array = zeros(numel(exp),1);
ShapeInserter = vision.ShapeInserter('LineWidth',lineWidth,'BorderColor','Custom','CustomBorderColor',lineColor);
imGTbox = step(ShapeInserter, imGT, int32(cat(2,fliplr(boxPose),fliplr(boxSize))));
imwrite(imGTbox,fullfile(figDir,figName,[imgName,'_GTbox','.png']));
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

fid = fopen([figName,'.tex'],'w');
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
fprintf(fid,'\\end{document}');

%--------------------------------------------------------------------------
% figure type 2. 
%
% ¦£------¦¤¦£------¦¤¦£------¦¤¦£------¦¤¦£------¦¤
% ¦¢      ¦¢¦¢      ¦¢¦¢      ¦¢¦¢      ¦¢¦¢      ¦¢
% ¦¢------¦¢¦¢------¦¢¦¢------¦¢¦¢------¦¢¦¢------¦¢
% ¦¢      ¦¢¦¢      ¦¢¦¢      ¦¢¦¢      ¦¢¦¢      ¦¢
% ¦¦------¦¥¦¦------¦¥¦¦------¦¥¦¦------¦¥¦¦------¦¥
%--------------------------------------------------------------------------
dataset = {'Set5'};
imgNum = 1;
boxSize = [60 60];
boxPose = [200 150];
lineWidth = 2;
lineColor = [255 0 0];
problem = 'SR';
sf = 3;
exp = {'HR','Bicubic','A+','SRCNN','RCN basic'};
figName = 'fig2';
figDir = 'figs';

numColumn = numel(exp);

if ~exist(fullfile(figDir,figName),'dir')
    mkdir(fullfile(figDir,figName));
end
datasetName = dataset{1};
SF = sf;
gtDir = fullfile('data',datasetName);
outDir = fullfile('data','result');
img_lst = dir(gtDir); img_lst = img_lst(3:end);
[~,imgName,imgExt] = fileparts(img_lst(imgNum).name);
imGT = imread(fullfile(gtDir, [imgName,imgExt]));
imGT = modcrop(imGT, SF);
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

fid = fopen([figName,'.tex'],'w');
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
fprintf(fid,'\\end{document}');

%--------------------------------------------------------------------------

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
else
    imGT = rgb2ycbcr(imGT);
    imGT = modcrop(imGT, SF);
    imSRcolor = cat(3,imSR(:,:,1),imGT(:,:,2),imGT(:,:,3));
    imSRcolor = ycbcr2rgb(imSRcolor);
end

% ToDo :
% Fig 1 : Good Qualitative result (zoomed)
% add recent method (from CVPR15, ICCV15 ...)
% Fig 2 : Good Qualitative result 
        