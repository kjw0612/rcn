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
%do.exp = {{'Bicubic', 'Bicubic'},{'SRCNN', 'SRCNN'},{'A+', 'A+'},{'RFL', 'RFL'}};
do.exp = {{'A+','A+'}};
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
% evalSetting(end+1) = evalSet('RCN mtest', 'RCN', 'Urban100', 3, 'best64.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Set5', 3, 'sf3/best_D15_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Set14', 3, 'sf3/best_D15_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'B100', 3, 'sf3/best_D15_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Urban100', 3, 'sf3/best_D15_F256.mat');

evalSetting(end+1) = evalSet('RCN', 'RCN', 'Set5', 4, 'sf4/best_D15_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Set14', 4, 'sf4/best_D15_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'B100', 4, 'sf4/best_D15_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Urban100', 4, 'sf4/best_D15_F256.mat');

evalSetting(end+1) = evalSet('RCN', 'RCN', 'Set5', 2, 'sf2/best_D15_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Set14', 2, 'sf2/best_D15_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'B100', 2, 'sf2/best_D15_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Urban100', 2, 'sf2/best_D15_F256.mat');

evalSetting(end+1) = evalSet('RCNd5', 'RCN', 'Set5', 3, 'sf3/best_D5_F256.mat');
evalSetting(end+1) = evalSet('RCNd10', 'RCN', 'Set5', 3, 'sf3/best_D10_F256.mat');
evalSetting(end+1) = evalSet('RCNd20', 'RCN', 'Set5', 3, 'sf3/best_D20_F256.mat');

evalSetting(end+1) = evalSet('RCNd5', 'RCN', 'Set14', 3, 'sf3/best_D5_F256.mat');
evalSetting(end+1) = evalSet('RCNd10', 'RCN', 'Set14', 3, 'sf3/best_D10_F256.mat');
evalSetting(end+1) = evalSet('RCNd20', 'RCN', 'Set14', 3, 'sf3/best_D20_F256.mat');

evalSetting(end+1) = evalSet('RCNd5', 'RCN', 'B100', 3, 'sf3/best_D5_F256.mat');
evalSetting(end+1) = evalSet('RCNd10', 'RCN', 'B100', 3, 'sf3/best_D10_F256.mat');
evalSetting(end+1) = evalSet('RCNd20', 'RCN', 'B100', 3, 'sf3/best_D20_F256.mat');

evalSetting(end+1) = evalSet('RCNd5', 'RCN', 'Urban100', 3, 'sf3/best_D5_F256.mat');
evalSetting(end+1) = evalSet('RCNd10', 'RCN', 'Urban100', 3, 'sf3/best_D10_F256.mat');
evalSetting(end+1) = evalSet('RCNd20', 'RCN', 'Urban100', 3, 'sf3/best_D20_F256.mat');

for d = [1 4 7 10 11]
    evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 2, 'sf2/best_D15_F256.mat', d);
    evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 3, 'sf3/best_D15_F256.mat', d);
    evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 4, 'sf4/best_D15_F256.mat', d);
end

evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 2, 'sf2/best_D15_F256.mat', [1:4]);
evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 3, 'sf3/best_D15_F256.mat', [1:4]);
evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 4, 'sf4/best_D15_F256.mat', [1:4]);

evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 2, 'sf2/best_D15_F256.mat', [1:7]);
evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 3, 'sf3/best_D15_F256.mat', [1:7]);
evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 4, 'sf4/best_D15_F256.mat', [1:7]);

evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 2, 'sf2/best_D15_F256.mat', [1:10]);
evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 3, 'sf3/best_D15_F256.mat', [1:10]);
evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 4, 'sf4/best_D15_F256.mat', [1:10]);

evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 2, 'sf2/best_D15_F256.mat', [1:11]);
evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 3, 'sf3/best_D15_F256.mat', [1:11]);
evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 4, 'sf4/best_D15_F256.mat', [1:11]);
% Setup outDir
outDir = 'data/result';
if ~exist('data/result', 'dir'), mkdir('data/result'); end

fileID = fopen('paper/rcn_eval_test.tex','w');
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
t1opts.sf = [2 3 4];
t1opts.exp = {'Bicubic','A+','SRCNN','RFL','SelfEx','RCN'};
t1opts.printTime = false;
t1opts.tableName = 'table_1';
t1opts.fid = fileID;%fopen([t1opts.tableName,'.tex'],'w');

%--------------------------------------------------------------------------
% table type 2.
%--------------------------------------------------------------------------
t2opts.dataset = {'Set5','Set14','B100','Urban100'};
t2opts.problem = 'SR';
t2opts.sf = [2 3 4];
t2opts.exp = {'Bicubic','A+','SRCNN','RFL','SelfEx','RCN'};
t2opts.printTime = false;
t2opts.tableName = 'table_2';
t2opts.fid = fileID;
%--------------------------------------------------------------------------
% for front explantory figure
%
%--------------------------------------------------------------------------
ffopts.problem = 'SR';
ffopts.dataset = 'B100';
ffopts.imgNum = '78004';
ffopts.boxSize = [60 60];
ffopts.boxPose = [];
ffopts.lineWidth = 4;
ffopts.lineColor = [255 0 0];
ffopts.sf = 3;
ffopts.exp = {'A+', 'SRCNN', 'SelfEx', 'RCN'};
ffopts.figName = 'figf';
ffopts.figDir = 'paper/figs';
ffopts.fid = fileID;
%--------------------------------------------------------------------------
% figure type 1. 
%
% t------tt------tt------tt------t
% | HR   |t------tt------tt------t
% |      |t------tt------tt------t
% t------tt------tt------tt------t
%--------------------------------------------------------------------------
f1opts.dataset = 'Urban100';
f1opts.imgNum = 96;
f1opts.boxSize = [60 100];
f1opts.boxPose = [];
f1opts.lineWidth = 4;
f1opts.lineColor = [255 0 0];
f1opts.problem = 'SR';
f1opts.sf = 3;
f1opts.exp = {'HR','A+','SRCNN','RFL','SelfEx','RCN'};
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
f2opts.boxSize = [];%if empty, box size = image size / 10
f2opts.boxPose = [];%if empty, it finds the best position.
f2opts.lineWidth = 4;
f2opts.lineColor = [255 0 0];
f2opts.problem = 'SR';
f2opts.sf = 3;
f2opts.exp = {'HR','A+','SRCNN','RFL','SelfEx','RCN'};
f2opts.figName = 'fig2';
f2opts.figDir = 'paper/figs';
f2opts.fid = fileID;

fSuppopts = f2opts;
fSuppopts.figName = 'fig3';
%--------------------------------------------------------------------------
% PSNR graph for different depths, residual etc.
%
%
%--------------------------------------------------------------------------

g1opts.dataset = {'Set5','Set14','B100','Urban100'}; %should be one string like dataset={'Set5'};
g1opts.sf = [3];
g1opts.exp = {'RCNd5','RCNd10','RCN','RCNd20'};
g1opts.graphSize = [];
g1opts.graphX = [5 10 15 20]-4;
g1opts.graphXname = {'1','6','11','16'};
g1opts.graphYname = [];
g1opts.graphName = 'graph1';
g1opts.graphDir = 'paper/figs';
g1opts.printOne = true;
g1opts.printLegend = true;
g1opts.fid = fileID;

%--------------------------------------------------------------------------
% De-ensembling effect.
% draw a table like (on Set5)
%     d1  d2  d3  d4  ... d12 || ensemble
% x2
% x3
% x4
%--------------------------------------------------------------------------
tiopts.dataset = {'Set5'};
tiopts.problem = 'SR';
tiopts.sf = [2 3 4];
tiopts.exp = {'RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCN'};
% 
%               'RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter', ...
%               'RCN'};
tiopts.interSetting = {1, 4, 7, 10, 11, [1:4], [1:7], [1:10], [1:11]};
tiopts.printTime = false;
tiopts.tableName = 'table_inter';
tiopts.fid = fileID;

%--------------------------------------------------------------------------

%%%% everything for paper

texPrefix(fileID);
% % makeTable1(t1opts);
% makeTable2(t2opts);
% % makeFigureFront(ffopts);
% f1opts.dataset = 'Urban100'; f1opts.imgNum = 96;
% makeFigure1(f1opts);
% f1opts.exp = {'HR','SRCNN','SelfEx','RCN'};
% makeFigure1(f1opts);
% f2opts.boxSize = [60 60];
% f2opts.dataset = 'Urban100'; f2opts.imgNum = 82;
% makeFigure2(f2opts);
% f2opts.dataset = 'Urban100'; f2opts.imgNum = 99;
% makeFigure2(f2opts);
% fSuppopts.boxSize = [90 90];
% makeFigureSupp(fSuppopts);
% clf;
% makeGraph(g1opts);
% g1opts.printOne = false;
makeTableInter(tiopts);
texSuffix(fileID);
fclose(fileID);


%--------------------------------------------------------------------------
%%%% for only figures in paper

fileID3 = fopen('paper/figuresOnly.tex','w');
texPrefix(fileID3);

% ffopts.fid = fileID3; 
% makeFigureFront(ffopts);

f1opts.fid = fileID3;
f1opts.dataset = 'Urban100'; f1opts.imgNum = 96;
makeFigure1(f1opts);

f1opts.exp = {'HR','SRCNN','SelfEx','RCN'};
makeFigure1(f1opts);

f2opts.fid = fileID3;
f2opts.boxSize = [60 60];
f2opts.dataset = 'Urban100'; f2opts.imgNum = 82;
makeFigure2(f2opts);

f2opts.dataset = 'Urban100'; f2opts.imgNum = 99;
makeFigure2(f2opts);
g1opts.fid = fileID3;
makeGraph(g1opts);
% tiopts.fid = fileID3;
% makeTableInter(tiopts);
texSuffix(fileID3);
fclose(fileID3);

%--------------------------------------------------------------------------
%%%% for supp. 

% cp = 0;
% 
% fileID2 = fopen('paper/supplebook.tex','w');
% texPrefix(fileID2);
% fSuppopts.fid = fileID2;
% fSuppopts.figName = 'figSup';
% dataset = {'Set5','Set14','B100','Urban100'};
% for d = 1:numel(dataset)
%     fSuppopts.dataset = dataset{d};
%     if strcmp(fSuppopts.dataset, 'Urban100')
%         fSuppopts.lineWidth = 8;
%         fSuppopts.boxSize = [90 90];
%     else
%         fSuppopts.lineWidth = 4;
%         fSuppopts.boxSize = [30 30];
%     end
%     for i = 1:numel(dir(fullfile('data',fSuppopts.dataset)))-2
%         fSuppopts.imgNum = i;
%         makeFigureSupp(fSuppopts);
%         cp = cp + 1;
%         if cp == 6, clearpage(fileID2); cp = 0; end;
%     end
% end
% 
% texSuffix(fileID2);
% fclose(fileID2);

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
fprintf(fid,'\\begin{table*}\n\\begin{center}\n');
fprintf(fid,'\\setlength{\\tabcolsep}{2pt}\n');
if numel(exp) >= 5
    fprintf(fid,'\\footnotesize\n');
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
    if strcmp('RCN', exp{indExp})
        fprintf(fid,[' & DRCN (Ours)']);
    else
        fprintf(fid,[' & ',exp{indExp}]);
    end
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
fprintf(fid,['\\caption{PSNR/SSIM for scale factor $\\times$',num2str(SF),' for ',datasetName, ... 
    '. {\\color{red}Red color} indicates the best performance and {\\color{blue}blue color} indicates the second best one.}\n']);
fprintf(fid,'\\end{center}\n\\end{table*}\n\n');  

function makeTable2(opts)

dataset = opts.dataset;
problem = opts.problem;
sf = opts.sf;
exp = opts.exp;
tableName = opts.tableName;
fid = opts.fid;
printTime = opts.printTime;

fprintf(fid,'\\begin{table*}\n\\begin{center}\n');
fprintf(fid,'\\setlength{\\tabcolsep}{2pt}\n');
if numel(exp) >= 5
    fprintf(fid,'\\footnotesize\n');
else
    fprintf(fid,'\\small\n');
end
fprintf(fid,'\\begin{tabular}{ |');
for indColumn = 1:numel(exp)+2
    fprintf(fid,' c |');
end
fprintf(fid,' }\n\\hline\n');
fprintf(fid,'\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Scale}');
for indExp = 1:numel(exp)
    if strcmp('RCN', exp{indExp})
        fprintf(fid,[' & DRCN (Ours)']);
    else
        fprintf(fid,[' & ',exp{indExp}]);
    end
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
fprintf(fid,'\\caption{Average PSNR/SSIM for scale factor ');
for i=1:numel(sf)
    if i < numel(sf)-1
        fprintf(fid,['$\\times$',num2str(sf(i)),', ']);
    elseif i == numel(sf)-1
        fprintf(fid,['$\\times$',num2str(sf(i)),' and ']);
    else
        fprintf(fid,['$\\times$',num2str(sf(i))]); 
    end
end
fprintf(fid,' on datasets ');
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
fprintf(fid,'\\end{center}\n\\end{table*}\n\n');  

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
if isa(imgNum,'string')
    imgName = imgNum;
else
    img_lst = dir(gtDir); img_lst = img_lst(3:end);
    [~,imgName,imgExt] = fileparts(img_lst(imgNum).name);
end
imGT = imread(fullfile(gtDir, [imgName,imgExt]));
imGT = modcrop(imGT, SF);
imSRCNN = []; imAplus = []; imRCN = [];
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
    if isempty(imAplus)
        imAplus = imSRCNN;
    end
    imGTs = shave(imGT,[SF SF]);
    imSRCNN = shave(imSRCNN,[SF SF]);
    imAplus = shave(imAplus,[SF SF]);
    imRCN = shave(imRCN,[SF SF]);
    [boxPose(1),boxPose(2)] = findBestPos(imGTs, imSRCNN, imAplus, imRCN, boxSize);
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
        if size(imGT,3) == 1, imGT = cat(3,imGT,imGT,imGT); end;
        imSRcolor = modcrop(imGT, SF);
    end
    imSRcolor = shave(imSRcolor,[SF SF]);
    subimSRcolor = imSRcolor(boxPose(1):boxPose(1)+boxSize(1)-1,boxPose(2):boxPose(2)+boxSize(2)-1,:);
    imwrite(subimSRcolor,fullfile(figDir,figName,[imgName,'_for_',figName,'_',expName,'.png']));
end
imGTbox = step(ShapeInserter, imGT, int32(cat(2,fliplr(boxPose),fliplr(boxSize))));
imwrite(imGTbox,fullfile(figDir,figName,[imgName,'_GTbox','.png']));

[~,maxPSNR] = max(PSNR_array,[],1); [~,maxSSIM] = max(SSIM_array,[],1);
[~,secmaxPSNR] = secmax(PSNR_array,1); [~,secmaxSSIM] = secmax(SSIM_array,1); 

fprintf(fid,'\\begin{figure*}\n');
fprintf(fid,'\\begin{adjustwidth}{0cm}{-0.5cm}\n');
fprintf(fid,'\\begin{center}\n');
fprintf(fid,'\\footnotesize\n');
fprintf(fid,'\\setlength{\\tabcolsep}{5pt}\n');
fprintf(fid,'\\begin{tabular}{ c');
for indColumn = 1:numColumn
    if numColumn == 2
        fprintf(fid,' C{4.5cm} ');
        widthVal = 0.26;
    else
        fprintf(fid,' C{3.5cm} ');
        widthVal = 0.20;
    end
end
fprintf(fid,' }\n');
fprintf(fid, ['\\multirow{4}{*}{\\graphicspath{{figs/',figName,'/}}\\includegraphics[width=0.27\\textwidth]{', ...
             [imgName,'_GTbox','.png'],'}}\n']);
indExp = 0;
indExp2 = 0;
for indRow = 1:4
    for indColumn = 1:numColumn
        if mod(indRow,2) == 1
            indExp = indExp + 1;
            fprintf(fid, ['& \\raisebox{-',num2str(boxSize(1)/4-2,'%.1f'),'ex} {\\graphicspath{{figs/',figName,'/}}\\includegraphics[width=',num2str(widthVal,'%.2f'),'\\textwidth]{', ...
                         [imgName,'_for_',figName,'_',exp{indExp},'.png'],'}}\\vspace{0.3ex}\n']);            
        else
            indExp2 = indExp2 + 1;
            if strcmp(exp{indExp2},'HR')
                fprintf(fid, ['& Original (PSNR, SSIM)']);
            else
                if strcmp(exp{indExp2},'RCN')
                    fprintf(fid, ['& DRCN (Ours) (']);
                else
                    fprintf(fid, ['& ',exp{indExp2},' (']);
                end
                if indExp2 == maxPSNR
                    fprintf(fid,['{\\color{red}{',num2str(PSNR_array(indExp2),'%.2f'),'}}, ']);
                    if indExp2 == maxSSIM
                        fprintf(fid,['{\\color{red}{',num2str(SSIM_array(indExp2),'%.4f'),'}})']);
                    elseif indExp2 == secmaxSSIM
                        fprintf(fid,['{\\color{blue}{',num2str(SSIM_array(indExp2),'%.4f'),'}})']);
                    else
                        fprintf(fid,[num2str(SSIM_array(indExp2),'%.4f'),')']);
                    end
                elseif indExp2 == secmaxPSNR
                    fprintf(fid,['{\\color{blue}{',num2str(PSNR_array(indExp2),'%.2f'),'}}, ']);
                    if indExp2 == maxSSIM
                        fprintf(fid,['{\\color{red}{',num2str(SSIM_array(indExp2),'%.4f'),'}})']);
                    elseif indExp2 == secmaxSSIM
                        fprintf(fid,['{\\color{blue}{',num2str(SSIM_array(indExp2),'%.4f'),'}})']);
                    else
                        fprintf(fid,[num2str(SSIM_array(indExp2),'%.4f'),')']);
                    end
                else
                    fprintf(fid,[num2str(PSNR_array(indExp2),'%.2f'),', ']);
                    if indExp2 == maxSSIM
                        fprintf(fid,['{\\color{red}{',num2str(SSIM_array(indExp2),'%.4f'),'}})']);
                    elseif indExp2 == secmaxSSIM
                        fprintf(fid,['{\\color{blue}{',num2str(SSIM_array(indExp2),'%.4f'),'}})']);
                    else
                        fprintf(fid,[num2str(SSIM_array(indExp2),'%.4f'),')']);
                    end
                end
            end
        end
    end    
    fprintf(fid, '\\\\\n');
end

fprintf(fid,'\\end{tabular}\n');
fprintf(fid,['\\caption{Super-resolution results of ``',setValidName(imgName,'_'),'"(',datasetName,') with scale factor $\\times$',num2str(SF),'. Our result is visually pleasing.}\n']);
fprintf(fid,'\\end{center}\n');
fprintf(fid,'\\end{adjustwidth}\n');
fprintf(fid,'\\end{figure*}\n\n');

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
if isa(imgNum,'string')
    imgName = imgNum;
else
    img_lst = dir(gtDir); img_lst = img_lst(3:end);
    [~,imgName,imgExt] = fileparts(img_lst(imgNum).name);
end
imGT = imread(fullfile(gtDir, [imgName,imgExt]));
imGT = modcrop(imGT, SF);

if isempty(boxSize)
    boxSize(1) = ceil(size(imGT,1)/10);
    boxSize(2) = ceil(size(imGT,2)/10);
end
imSRCNN = []; imAplus = []; imRCN = [];
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
    if isempty(imAplus)
        imAplus = imSRCNN;
    end
    imGTs = shave(imGT,[SF SF]);
    imSRCNN = shave(imSRCNN,[SF SF]);
    imAplus = shave(imAplus,[SF SF]);
    imRCN = shave(imRCN,[SF SF]);
    [boxPose(1),boxPose(2)] = findBestPos(imGTs, imSRCNN, imAplus, imRCN, boxSize);
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
        if size(imGT,3) == 1, imGT = cat(3,imGT,imGT,imGT); end;
        imSRcolor = modcrop(imGT, SF);
    end
    imSRcolor = shave(imSRcolor,[SF SF]); % for methods like A+, not predicting boundary.
    ShapeInserter = vision.ShapeInserter('LineWidth',lineWidth,'BorderColor','Custom','CustomBorderColor',lineColor);
    boximSRcolor = step(ShapeInserter, imSRcolor, int32(cat(2,fliplr(boxPose),fliplr(boxSize))));
    subimSRcolor = imSRcolor(boxPose(1):boxPose(1)+boxSize(1)-1,boxPose(2):boxPose(2)+boxSize(2)-1,:);
    subimSRcolor = imresize(subimSRcolor, [NaN,size(boximSRcolor,2)]);
    ShapeInserter = vision.ShapeInserter('LineWidth',lineWidth*SF,'BorderColor','Custom','CustomBorderColor',lineColor);
    subimSRcolor = step(ShapeInserter, subimSRcolor, int32([1,1,size(subimSRcolor,2),size(subimSRcolor,1)]));
    catimSRcolor = cat(1,boximSRcolor,subimSRcolor);
    imwrite(catimSRcolor,fullfile(figDir,figName,[imgName,'_for_',figName,'_',expName,'.png']));
end

[~,maxPSNR] = max(PSNR_array,[],1); [~,maxSSIM] = max(SSIM_array,[],1);
[~,secmaxPSNR] = secmax(PSNR_array,1); [~,secmaxSSIM] = secmax(SSIM_array,1); 

fprintf(fid,'\\begin{figure*}\n');
fprintf(fid,'\\begin{adjustwidth}{0.5cm}{0.5cm}\n');
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
        fprintf(fid, ['{\\graphicspath{{figs/',figName,'/}}\\includegraphics[width=',num2str(0.93/numel(exp),'%.2f'),'\\textwidth]{', ...
                     [imgName,'_for_',figName,'_',exp{indColumn},'.png'],'}}\n']);            
    else
        fprintf(fid, ['& {\\graphicspath{{figs/',figName,'/}}\\includegraphics[width=',num2str(0.93/numel(exp),'%.2f'),'\\textwidth]{', ...
                     [imgName,'_for_',figName,'_',exp{indColumn},'.png'],'}}\n']);            
    end   
end
fprintf(fid, '\\\\\n');
for indColumn = 1:numColumn
    if strcmp(exp{indColumn},'HR')
        fprintf(fid, 'Original');
    elseif strcmp(exp{indColumn},'RCN')
        fprintf(fid, ['& DRCN (Ours)']);
    else
        fprintf(fid, ['& ',exp{indColumn}]);
    end
end
fprintf(fid, '\\\\\n');
for indColumn = 1:numColumn
    if strcmp(exp{indColumn},'HR')
        fprintf(fid, '(PSNR, SSIM)');
    else
        if indColumn == maxPSNR
            fprintf(fid,['& ({\\color{red}{',num2str(PSNR_array(indColumn),'%.2f'),'}}, ']);
            if indColumn == maxSSIM
                fprintf(fid,['{\\color{red}{',num2str(SSIM_array(indColumn),'%.4f'),'}})']);
            elseif indColumn == secmaxSSIM
                fprintf(fid,['{\\color{blue}{',num2str(SSIM_array(indColumn),'%.4f'),'}})']);
            else
                fprintf(fid,[num2str(SSIM_array(indColumn),'%.4f'),')']);
            end
        elseif indColumn == secmaxPSNR
            fprintf(fid,['& ({\\color{blue}{',num2str(PSNR_array(indColumn),'%.2f'),'}}, ']);
            if indColumn == maxSSIM
                fprintf(fid,['{\\color{red}{',num2str(SSIM_array(indColumn),'%.4f'),'}})']);
            elseif indColumn == secmaxSSIM
                fprintf(fid,['{\\color{blue}{',num2str(SSIM_array(indColumn),'%.4f'),'}})']);
            else
                fprintf(fid,[num2str(SSIM_array(indColumn),'%.4f'),')']);
            end
        else
            fprintf(fid,['& (',num2str(PSNR_array(indColumn),'%.2f'),', ']);
            if indColumn == maxSSIM
                fprintf(fid,['{\\color{red}{',num2str(SSIM_array(indColumn),'%.4f'),'}})']);
            elseif indColumn == secmaxSSIM
                fprintf(fid,['{\\color{blue}{',num2str(SSIM_array(indColumn),'%.4f'),'}})']);
            else
                fprintf(fid,[num2str(SSIM_array(indColumn),'%.4f'),')']);
            end
        end        
    end
end
fprintf(fid, '\\\\\n');
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,['\\caption{Super-resolution results of ``',setValidName(imgName,'_'),'"(',datasetName,') with scale factor $\\times$',num2str(SF),'. Our result is visually pleasing.}\n']);
fprintf(fid,'\\end{center}\n');
fprintf(fid,'\\end{adjustwidth}\n');
fprintf(fid,'\\end{figure*}\n\n');

function makeFigureSupp(opts)

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
if isa(imgNum,'string')
    imgName = imgNum;
else
    img_lst = dir(gtDir); img_lst = img_lst(3:end);
    [~,imgName,imgExt] = fileparts(img_lst(imgNum).name);
end
imGT = imread(fullfile(gtDir, [imgName,imgExt]));
imGT = modcrop(imGT, SF);
if size(imGT,1) > size(imGT,2)
    imLong = true;
else
    imLong = false;
end


if isempty(boxSize)
    boxSize(1) = ceil(size(imGT,1)/10);
    boxSize(2) = ceil(size(imGT,2)/10);
end

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
    imGTs = shave(imGT,[SF SF]);
    imSRCNN = shave(imSRCNN,[SF SF]);
    imAplus = shave(imAplus,[SF SF]);
    imRCN = shave(imRCN,[SF SF]);
    [boxPose(1),boxPose(2)] = findBestPos(imGTs, imSRCNN, imAplus, imRCN, boxSize);
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
        if size(imGT,3) == 1, imGT = cat(3,imGT,imGT,imGT); end;
        imSRcolor = modcrop(imGT, SF);
    end
    imSRcolor = shave(imSRcolor,[SF SF]); % for methods like A+, not predicting boundary.    
    ShapeInserter = vision.ShapeInserter('LineWidth',lineWidth,'BorderColor','Custom','CustomBorderColor',lineColor);
    boximSRcolor = step(ShapeInserter, imSRcolor, int32(cat(2,fliplr(boxPose),fliplr(boxSize))));
    subimSRcolor = imSRcolor(boxPose(1):boxPose(1)+boxSize(1)-1,boxPose(2):boxPose(2)+boxSize(2)-1,:);
    subimSRcolor = imresize(subimSRcolor, [NaN,size(boximSRcolor,2)]);
    ShapeInserter = vision.ShapeInserter('LineWidth',lineWidth*SF,'BorderColor','Custom','CustomBorderColor',lineColor);
    subimSRcolor = step(ShapeInserter, subimSRcolor, int32([1,1,size(subimSRcolor,2),size(subimSRcolor,1)]));
    catimSRcolor = cat(1,boximSRcolor,subimSRcolor);
    imwrite(catimSRcolor,fullfile(figDir,figName,[imgName,'_for_',figName,'_',expName,'.png']));
end

fprintf(fid,'\\begin{figure*}\n');
fprintf(fid,'\\begin{adjustwidth}{0.5cm}{0.5cm}\n');
fprintf(fid,'\\begin{center}\n');
%fprintf(fid,'\\small\n');
fprintf(fid,'\\setlength{\\tabcolsep}{3pt}\n');
fprintf(fid,'\\begin{tabular}{ ');
for indColumn = 1:numColumn
    fprintf(fid,' c ');
end
fprintf(fid,' }\n');
indExp = 0;
indExp2 = 0;
indExp3 = 0;
for indRow = 1:2
    for indColumn = 1:numColumn
        indExp = indExp + 1;
        if imLong         
            if indColumn == 1
                fprintf(fid, ['{\\graphicspath{{figs/',figName,'/}}\\includegraphics[height=',num2str(0.85/2,'%.2f'),'\\textheight]{', ...
                             [imgName,'_for_',figName,'_',exp{indExp},'.png'],'}}\n']);

            else
                fprintf(fid, ['& {\\graphicspath{{figs/',figName,'/}}\\includegraphics[height=',num2str(0.85/2,'%.2f'),'\\textheight]{', ...
                             [imgName,'_for_',figName,'_',exp{indExp},'.png'],'}}\n']);       
            end
        else
            if indColumn == 1
                fprintf(fid, ['{\\graphicspath{{figs/',figName,'/}}\\includegraphics[width=',num2str(0.85/numColumn,'%.2f'),'\\textwidth]{', ...
                             [imgName,'_for_',figName,'_',exp{indExp},'.png'],'}}\n']);
            else
                fprintf(fid, ['& {\\graphicspath{{figs/',figName,'/}}\\includegraphics[width=',num2str(0.85/numColumn,'%.2f'),'\\textwidth]{', ...
                             [imgName,'_for_',figName,'_',exp{indExp},'.png'],'}}\n']);       
            end     
        end
    end    
    fprintf(fid, '\\\\\n');
    for indColumn = 1:numColumn
        indExp2 = indExp2 + 1;
        if strcmp(exp{indExp2},'HR')
            fprintf(fid, 'Original');
        elseif indColumn == 1
            fprintf(fid, exp{indExp2});
        else
            fprintf(fid, ['& ',exp{indExp2}]);
        end
    end
    fprintf(fid, '\\\\\n');
    for indColumn = 1:numColumn
        indExp3 = indExp3 + 1;
        if strcmp(exp{indExp3},'HR')
            fprintf(fid, '(PSNR, SSIM)');
        elseif indColumn == 1
            fprintf(fid, ['(',num2str(PSNR_array(indExp3),'%.2f'),', ',num2str(SSIM_array(indExp3),'%.4f'),')']);
        else
            fprintf(fid, ['& (',num2str(PSNR_array(indExp3),'%.2f'),', ',num2str(SSIM_array(indExp3),'%.4f'),')']);
        end
    end
    fprintf(fid, '\\\\\n');
end
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,['\\caption{Super-resolution results of ``',setValidName(imgName,'_'),'"(',datasetName,') with scale factor $\\times$',num2str(SF),'. Our result is visually pleasing.}\n']);
fprintf(fid,'\\end{center}\n');
fprintf(fid,'\\end{adjustwidth}\n');
fprintf(fid,'\\end{figure*}\n\n');

function makeFigureFront(opts)

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
if isa(imgNum,'char')
    imgNum = findImgNum(dataset, imgNum);
end

img_lst = dir(gtDir); img_lst = img_lst(3:end);
numImg = numel(img_lst);
[~,imgName,imgExt] = fileparts(img_lst(imgNum).name);

imGT = imread(fullfile(gtDir, [imgName,imgExt]));
imGT = modcrop(imGT, SF);
imSRCNN = []; imAplus = []; imRCN = [];
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
    if isempty(imAplus)
        imAplus = imSRCNN;
    end
    imGTs = shave(imGT,[SF SF]);
    imSRCNNs = shave(imSRCNN,[SF SF]);
    imApluss = shave(imAplus,[SF SF]);
    imRCNs = shave(imRCN,[SF SF]);
    [boxPose(1),boxPose(2)] = findBestPos(imGTs, imSRCNNs, imApluss, imRCNs, boxSize);
end
SSIM_table = zeros(numImg, numel(exp));
imgNames = cell(numImg,1);

for indImg = 1:numImg
    [~,imgNames{indImg},imgExt] = fileparts(img_lst(indImg).name);
    imGT_ = imread(fullfile(gtDir, [imgNames{indImg},imgExt]));            

    for indExp = 1:numel(exp)
        expName = exp{indExp};
        outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
        imSR = imread(fullfile(outDir, outRoute, [imgNames{indImg},'.png']));
        [~, ssim] = compute_diff(imGT_, imSR, SF);
        SSIM_table(indImg, indExp) = ssim;
    end
end

avgSSIM = mean(SSIM_table,1);
% stdSSIM = std(SSIM_table,0,1);
fHand = figure;
aHand = axes('parent', fHand);
hold(aHand, 'on');
colors = summer(numel(avgSSIM));
for i = 1:numel(avgSSIM)
    bar(i, avgSSIM(i), 'parent', aHand, 'facecolor', colors(i,:), 'barwidth', 0.8);
end
% err = errorbar(avgSSIM, stdSSIM,'r'); set(err,'linestyle','none')
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) 350 400]); %<- Set size
set(gca, 'FontSize', 15, 'LineWidth', 1.0);%, 'FontWeight', 'bold'); %<- Set properties
grid on; 
legend('A+', 'SRCNN', 'SelfEx', 'RCN(Ours)','Location','northwest');
set(gca, 'XTick', [], 'XTickLabel', []);
ylim([min(avgSSIM)-0.001 max(avgSSIM)+0.001]);
set(gcf, 'PaperPosition', [0 0 5 5]); %Position plot at left hand corner with width 5 and height 5.
tightfig;
print(fullfile('paper','figs','figf','frontfig'), '-dpdf', '-r600');

ShapeInserter = vision.ShapeInserter('LineWidth',lineWidth,'BorderColor','Custom','CustomBorderColor',lineColor);
imGTbox = step(ShapeInserter, imGT, int32(cat(2,fliplr(boxPose),fliplr(boxSize))));
imSRCNNcolor = colorize(imGT, imSRCNN, SF);
imRCNcolor = colorize(imGT, imRCN, SF);
imSRCNNcolor = shave(imSRCNNcolor,[SF SF]); imRCNcolor = shave(imRCNcolor, [SF, SF]);
subimSRCNNcolor = imSRCNNcolor(boxPose(1):boxPose(1)+boxSize(1)-1,boxPose(2):boxPose(2)+boxSize(2)-1,:);
subimRCNcolor = imRCNcolor(boxPose(1):boxPose(1)+boxSize(1)-1,boxPose(2):boxPose(2)+boxSize(2)-1,:);
imGTbox = modcrop(imGTbox,2);
subimSRCNNcolor = imresize(subimSRCNNcolor,[NaN ceil(size(imGTbox,2)/2)]);
subimRCNcolor = imresize(subimRCNcolor, [NaN floor(size(imGTbox,2)/2)]);
bot = cat(2,subimSRCNNcolor, subimRCNcolor);
whole = cat(1, imGTbox, bot);
imwrite(whole,fullfile(figDir,figName,[imgName,'_frontfig','.png']));


fprintf(fid,'\\begin{figure}\n');
fprintf(fid,'\\begin{adjustwidth}{0cm}{-0.5cm}\n');
fprintf(fid,'\\centering\n');
fprintf(fid,'\\begin{subfigure}{0.12\textwidth}\n');
fprintf(fid,'\\centering\n');
fprintf(fid,'{\\graphicspath{{figs/figf/}}\\includegraphics[height=1.5in]{frontfig.pdf}}\n');
fprintf(fid,'\\end{subfigure}\n');
fprintf(fid,'\\hfill\n');
fprintf(fid,'\\begin{subfigure}{0.35\\textwidth}\n');
fprintf(fid,'\\centering\n');
fprintf(fid,['{\\graphicspath{{figs/figf/}}\\includegraphics[height=1.5in]{',imgName,'_frontfig.png}}\n']);
fprintf(fid,'\\end{subfigure}\n');
fprintf(fid,'\\end{adjustwidth}\n');
fprintf(fid,'\\end{figure}\n\n');

function makeTableInter(opts)

dataset = opts.dataset;
problem = opts.problem;
sf = opts.sf;
exp = opts.exp;
tableName = opts.tableName;
interSetting = opts.interSetting;
fid = opts.fid;
printTime = opts.printTime;

fprintf(fid,'\\begin{table*}\n\\begin{center}\n');
fprintf(fid,'\\setlength{\\tabcolsep}{5pt}\n');
if numel(exp) >= 5
    fprintf(fid,'\\footnotesize\n');
else
    fprintf(fid,'\\small\n');
end
fprintf(fid,'\\begin{tabular}{ |');
for indColumn = 1:numel(exp)+1
    fprintf(fid,' c |');
end
fprintf(fid,' }\n\\hline\n');
fprintf(fid,'Scale');
for indExp = 1:numel(exp)
    if strcmp('RCN', exp{indExp})
        fprintf(fid,[' & Ensemble']);
    else
        if numel(interSetting{indExp}) > 1
            fprintf(fid,[' & $\\sum\\limits_{d=1}^{',num2str(interSetting{indExp}(end)),'}$ Output $d$']);
        else
            fprintf(fid,[' & Output ',num2str(interSetting{indExp})]);
        end
    end
end
% fprintf(fid,'\\\\\n &');
% for indExp = 1:numel(exp)
%     if printTime
%         fprintf(fid, ' & PSNR/SSIM/time');
%     else
%         fprintf(fid, ' & PSNR/SSIM');
%     end
% end
fprintf(fid,'\\\\\n\\hline\n');

for indDataset = 1:numel(dataset)
    fprintf(fid,'\\hline\n');
    datasetName = dataset{indDataset};
    gtDir = fullfile('data',datasetName);
    outDir = fullfile('data','result');
    img_lst = dir(gtDir); img_lst = img_lst(3:end);
    numImg = numel(img_lst);    

%     fprintf(fid, ['\\multirow{',num2str(numel(sf)),'}{*}{',setValidName(datasetName,'_'),'}']);
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
                if strcmp(expName,'RCNInter')
                    outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF),'_d',num2str(interSetting{indExp})]);
                else
                    outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
                end
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
                
        fprintf(fid,['$\\times$', num2str(SF)]);            
        for indExp = 1:numel(exp)
            if indExp == maxAvgPSNR
                fprintf(fid, [' & {\\color{red}',num2str(avgPSNR(1, indExp),'%.2f'),'}']);
%                 if indExp == maxAvgSSIM
%                     fprintf(fid, ['/{\\color{red}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
%                 elseif indExp == secmaxAvgSSIM
%                     fprintf(fid, ['/{\\color{blue}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
%                 else
%                     fprintf(fid, ['/',num2str(avgSSIM(1, indExp),'%.4f')]);
%                 end
%                 if printTime
%                     fprintf(fid, ['/',num2str(avgTIME(1, indExp),'%.2f')]);
%                 end
%             elseif indExp == secmaxAvgPSNR
%                 fprintf(fid, [' & {\\color{blue}',num2str(avgPSNR(1, indExp),'%.2f'),'}']);
%                 if indExp == maxAvgSSIM
%                     fprintf(fid, ['/{\\color{red}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
%                 elseif indExp == secmaxAvgSSIM
%                     fprintf(fid, ['/{\\color{blue}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
%                 else
%                     fprintf(fid, ['/',num2str(avgSSIM(1, indExp),'%.4f')]);
%                 end
%                 if printTime
%                     fprintf(fid, ['/',num2str(avgTIME(1, indExp),'%.2f')]);
%                 end
            else
                fprintf(fid, [' & ',num2str(avgPSNR(1, indExp),'%.2f')]);
%                 if indExp == maxAvgSSIM
%                     fprintf(fid, ['/{\\color{red}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
%                 elseif indExp == secmaxAvgSSIM
%                     fprintf(fid, ['/{\\color{blue}',num2str(avgSSIM(1, indExp),'%.4f'),'}']);
%                 else
%                     fprintf(fid, ['/',num2str(avgSSIM(1, indExp),'%.4f')]);
%                 end
%                 if printTime
%                     fprintf(fid, ['/',num2str(avgTIME(1, indExp),'%.2f')]);
%                 end
            end
        end
        fprintf(fid, '\\\\\n');        
    end
    fprintf(fid,'\\hline\n');
end
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,['\\caption{Experiment on the effect of ensemble.',...
    ' Quantitative evaluation (PSNR) on dataset ‘Set5’ is provided for scale factors 2,3 and 4. {\\color{red}Red color} indicates the best performance.}\n']);
fprintf(fid,'\\end{center}\n\\end{table*}\n\n');  

function makeGraph(opts)

dataset = opts.dataset; %should be one string like dataset={'Set5'};
sf = opts.sf;
exp = opts.exp;
graphSize = opts.graphSize;
graphX = opts.graphX;
graphXname = opts.graphXname;
graphYname = opts.graphYname;
graphName = opts.graphName;
graphDir = opts.graphDir;
printOne = opts.printOne;
printLegend = opts.printLegend;
fid = opts.fid;

if ~exist(fullfile(graphDir,graphName),'dir')
    mkdir(fullfile(graphDir,graphName));
end

if isempty(graphSize)
    graphSize = [400 400];
end

PSNR_table = cell(numel(dataset),numel(sf));
SSIM_table = cell(numel(dataset),numel(sf));

for indDataset = 1:numel(dataset)
    datasetName = dataset{indDataset};
    gtDir = fullfile('data',datasetName);
    outDir = fullfile('data','result');
    img_lst = dir(gtDir); img_lst = img_lst(3:end);
    numImg = numel(img_lst);

    for indSF = 1:numel(sf)
        SF = sf(indSF);
        PSNR_table{indSF} = zeros(numImg, numel(exp));
        SSIM_table{indSF} = zeros(numImg, numel(exp));
        imgNames = cell(numImg,1);

        for indImg = 1:numImg
            [~,imgNames{indImg},imgExt] = fileparts(img_lst(indImg).name);
            imGT = imread(fullfile(gtDir, [imgNames{indImg},imgExt]));            

            for indExp = 1:numel(exp)
                expName = exp{indExp};
                outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
                imSR = imread(fullfile(outDir, outRoute, [imgNames{indImg},'.png']));
                %elapsedTime = load(fullfile(outDir, outRoute, 'elapsed_time.mat'));

                [psnr, ssim] = compute_diff(imGT, imSR, SF);
                PSNR_table{indDataset,indSF}(indImg, indExp) = psnr;
                SSIM_table{indDataset,indSF}(indImg, indExp) = ssim;
            end
        end
    end
end
if printOne
    for indDataset = 1:numel(dataset)
        for indSF = 1:numel(sf) %for each sf, make figs
            lw =1.5;
            pos = get(gcf, 'Position');
            set(gcf, 'Position', [pos(1) pos(2) graphSize(1), graphSize(2)]); %<- Set size
            ax = gca; % current axes
            plot(graphX, mean(PSNR_table{indDataset,indSF},1), 'LineWidth', lw);
            xlim([min(graphX),max(graphX)]);
            ax.XTick = graphX;
            ax.XTickLabel = graphXname;
            ax.FontSize = 18;
            ax.LineWidth = lw;
            hold on;
            %plot(5:20, psnr_bicubs', '--','LineWidth', lw); 
            xlabel('Recursion');
            ylabel('PSNR (dB)');
            grid on;
            set(gcf, 'PaperPosition', [0, 0, 5, 5]);
            legendInfo{indDataset,indSF} = ['on ',dataset{indDataset},' \times',num2str(sf(indSF))];
            if printOne && indSF==numel(sf) && numel(sf)~=1            
                legend(legendInfo, 'Location', 'east'); 
            end;
            tightfig;
            print(fullfile('paper','figs',graphName,'graphOne'), '-dpdf', '-r600');
        end
    end
else
    for indSF = 1:numel(sf)
        clf;
        for indDataset = 1:numel(dataset)
            lw =1.5;
            pos = get(gcf, 'Position');
            set(gcf, 'Position', [pos(1) pos(2) graphSize(1), graphSize(2)]); %<- Set size
            ax = gca; % current axes
            plot(graphX, mean(PSNR_table{indDataset,indSF},1), 'LineWidth', lw);
            xlim([min(graphX),max(graphX)]);
            ax.XTick = graphX;
            ax.XTickLabel = graphXname;
            ax.FontSize = 18;
            ax.LineWidth = lw;
            %ax.FontWeight = 'bold';
            xlabel('Recursion');
            ylabel('PSNR (dB)');
            grid on;
            set(gcf, 'PaperPosition', [0, 0, 5, 5]);
            tightfig;        
        end
        print(fullfile('paper','figs',graphName,['graph',num2str(sf(indSF))]), '-dpdf', '-r600');
    end
end

if printOne
    fprintf(fid,'\\begin{figure}\n');
    fprintf(fid,'\\begin{adjustwidth}{0cm}{-0.0cm}\n');
    fprintf(fid,'\\centering\n');
    fprintf(fid,['{\\graphicspath{{figs/',graphName,'/}}\\includegraphics[height=4.8cm]{graphOne.pdf}}\n']);
    fprintf(fid,'\\caption{Recursion versus Performance for the scale factor ');
    for i=1:numel(sf)
        if i < numel(sf)-1
            fprintf(fid,['$\\times$',num2str(sf(i)),', ']);
        elseif i == numel(sf)-1
            fprintf(fid,['$\\times$',num2str(sf(i)),' and ']);
        else
            fprintf(fid,['$\\times$',num2str(sf(i))]); 
        end
    end
    if numel(dataset) == 1
        fprintf(fid,[' on the dataset ']);
    else
        fprintf(fid,[' on datasets \\textit{']);
    for i=1:numel(dataset)
        if i < numel(dataset)-1
            fprintf(fid,[dataset{i},', ']);
        elseif i == numel(dataset)-1
            fprintf(fid,[dataset{i},', ']);
        else
            fprintf(fid,[dataset{i},'}.\n']);
        end
    end
    fprintf(fid,'More recursions yielding larger receptive fields lead to better performances. (Graph {\\color{red}not complete} yet)\n');
    fprintf(fid,'\\end{adjustwidth}\n');        
    fprintf(fid,'\\end{figure}\n\n');
else
    fprintf(fid,'\\begin{figure*}\n');
    fprintf(fid,'\\begin{adjustwidth}{0cm}{-0.0cm}\n');
    fprintf(fid,'\\centering\n');
    for indSF = 1:numel(sf)
        fprintf(fid,'\\begin{subfigure}{0.3\\textwidth}\n');
        fprintf(fid,'\\centering\n');
        fprintf(fid,['{\\graphicspath{{figs/',graphName,'/}}\\includegraphics[height=4.8cm]{graph',...
                     num2str(sf(indSF)),'.pdf}}\n']);
        fprintf(fid,['\\caption{Test Scale Factor ',num2str(sf(indSF)),'}\n']);
        fprintf(fid,'\\end{subfigure}%%\n');
    end
    fprintf(fid,'\\caption{Depth vs Performance}');
    fprintf(fid,'\\end{adjustwidth}\n');
    fprintf(fid,'\\end{figure*}\n\n');
end


function texPrefix(fid)

fprintf(fid,'\\documentclass[10pt,twocolumn,letterpaper]{article}\n\n');
fprintf(fid,'\\usepackage{cvpr}\n');
fprintf(fid,'\\usepackage{times}\n');
fprintf(fid,'\\usepackage{epsfig}\n');
fprintf(fid,'\\usepackage{amsmath}\n');
fprintf(fid,'\\usepackage{amssymb}\n');
fprintf(fid,'\\usepackage{subcaption}\n');
fprintf(fid,'\\usepackage{multirow}\n');
fprintf(fid,'\\usepackage{color}\n');
fprintf(fid,'\\usepackage{graphicx}\n');
fprintf(fid,'\\usepackage[space]{grffile}\n');
fprintf(fid,'\\usepackage{array}\n');
fprintf(fid,'\\newcolumntype{C}[1]{>{\\centering\\arraybackslash}p{#1}}\n');
fprintf(fid,'\\usepackage{chngpage}\n\n');

fprintf(fid,'%% Include other packages here, before hyperref.\n');
fprintf(fid,'%% If you comment hyperref and then uncomment it, you should delete\n');
fprintf(fid,'%% egpaper.aux before re-running latex.  (Or just hit ''q'' on the first latex\n');
fprintf(fid,'%% run, let it finish, and you should be clear).\n\n');
fprintf(fid,'\\usepackage[pagebackref=true,breaklinks=true,letterpaper=true,colorlinks,bookmarks=false]{hyperref}\n');
fprintf(fid,'\\usepackage[font=small,labelfont=bf,tableposition=top]{caption}\n\n');

fprintf(fid,'%%\\cvprfinalcopy %% *** Uncomment this line for the final submission\n');
fprintf(fid,'\\def\\cvprPaperID{****} %% *** Enter the CVPR Paper ID here\n');
fprintf(fid,'\\def\\httilde{\\mbox{\\tt\\raisebox{-.5ex}{\\symbol{126}}}}\n');
fprintf(fid,'%% Pages are numbered in submission mode, and unnumbered in camera-ready\n');
fprintf(fid,'\\ifcvprfinal\\pagestyle{empty}\\fi\n\n');

fprintf(fid,'\\begin{document}\n\n');

function texSuffix(fid)
fprintf(fid,'{\\small\n');
fprintf(fid,'  \\bibliographystyle{ieee}\n');
fprintf(fid,'  \\bibliography{RCN}\n}\n');
fprintf(fid,'\\end{document}');

function clearpage(fid)
fprintf(fid,'\\clearpage\n\n');

function validName = setValidName(name, exp)
ind = regexp(name,exp);
if ind > 0
    validName = name(1:ind-1);
else
    validName = name;
end
    
function [sec,seci] = secmax(A,dim)
maxIndMatrix = bsxfun(@eq,A,max(A,[],dim));
A(maxIndMatrix) = -Inf;
[sec,seci] = max(A,[],dim);

function imSRcolor = colorize(imGT, imSR, SF)
if size(imGT,3) < 1
    imSRcolor = cat(3,imSR,imSR,imSR);
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
        val1 = ssim1c(x+boxSize(1)-1,y+boxSize(2)-1) - ssim1c(x+boxSize(1)-1,y-1) - ssim1c(x-1,y+boxSize(2)-1) + ssim1c(x-1,y-1);
        val2 = ssim2c(x+boxSize(1)-1,y+boxSize(2)-1) - ssim2c(x+boxSize(1)-1,y-1) - ssim2c(x-1,y+boxSize(2)-1) + ssim2c(x-1,y-1);
        val3 = ssim3c(x+boxSize(1)-1,y+boxSize(2)-1) - ssim3c(x+boxSize(1)-1,y-1) - ssim3c(x-1,y+boxSize(2)-1) + ssim3c(x-1,y-1);
        if log(val1)-log(max(val2,val3)) > max_val
            max_val= log(val1)-log(max(val2,val3));
            maxX = x;
            maxY = y;
        end
    end
end

function imgNum = findImgNum(dataset, imgName)
img_lst = dir(fullfile('data',dataset));
img_lst = img_lst(3:end);
for i = 1:numel(img_lst)
    [~,name,~] = fileparts(img_lst(i).name);
    if strcmp(imgName, name)
        imgNum = i;
        break;
    end
end
        