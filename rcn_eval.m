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
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Set5', 3, 'sf3/best_D20_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Set14', 3, 'sf3/best_D20_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'B100', 3, 'sf3/best_D20_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Urban100', 3, 'sf3/best_D20_F256.mat');

evalSetting(end+1) = evalSet('RCN', 'RCN', 'Set5', 4, 'sf4/best_D20_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Set14', 4, 'sf4/best_D20_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'B100', 4, 'sf4/best_D20_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Urban100', 4, 'sf4/best_D20_F256.mat');

evalSetting(end+1) = evalSet('RCN', 'RCN', 'Set5', 2, 'sf2/best_D20_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Set14', 2, 'sf2/best_D20_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'B100', 2, 'sf2/best_D20_F256.mat');
evalSetting(end+1) = evalSet('RCN', 'RCN', 'Urban100', 2, 'sf2/best_D20_F256.mat');

evalSetting(end+1) = evalSet('RCNd5', 'RCN', 'Set5', 3, 'sf3/best_D5_F256.mat');
evalSetting(end+1) = evalSet('RCNd10', 'RCN', 'Set5', 3, 'sf3/best_D10_F256.mat');
evalSetting(end+1) = evalSet('RCNd15', 'RCN', 'Set5', 3, 'sf3/best_D15_F256.mat');

evalSetting(end+1) = evalSet('RCNd5', 'RCN', 'Set14', 3, 'sf3/best_D5_F256.mat');
evalSetting(end+1) = evalSet('RCNd10', 'RCN', 'Set14', 3, 'sf3/best_D10_F256.mat');
evalSetting(end+1) = evalSet('RCNd15', 'RCN', 'Set14', 3, 'sf3/best_D15_F256.mat');

evalSetting(end+1) = evalSet('RCNd5', 'RCN', 'B100', 3, 'sf3/best_D5_F256.mat');
evalSetting(end+1) = evalSet('RCNd10', 'RCN', 'B100', 3, 'sf3/best_D10_F256.mat');
evalSetting(end+1) = evalSet('RCNd15', 'RCN', 'B100', 3, 'sf3/best_D15_F256.mat');

evalSetting(end+1) = evalSet('RCNd5', 'RCN', 'Urban100', 3, 'sf3/best_D5_F256.mat');
evalSetting(end+1) = evalSet('RCNd10', 'RCN', 'Urban100', 3, 'sf3/best_D10_F256.mat');
evalSetting(end+1) = evalSet('RCNd15', 'RCN', 'Urban100', 3, 'sf3/best_D15_F256.mat');

% for d = [1 3 5 7 9 11 13 15]
%     for n = [0 1 2 3 4]
%         evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 2, 'sf2/best_D20_F256.mat', [1 d n]);
%         evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 3, 'sf3/best_D20_F256.mat', [1 d n]);
%         evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 4, 'sf4/best_D20_F256.mat', [1 d n]);
%     end
% end

for d = [1:16]
    evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 2, 'sf2/best_D20_F256.mat', [d d 4]);
    evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 3, 'sf3/best_D20_F256.mat', [d d 4]);
    evalSetting(end+1) = evalSet('RCNInter', 'RCNInter', 'Set5', 4, 'sf4/best_D20_F256.mat', [d d 4]);
end

evalSetting(end+1) = evalSet('VDSR', 'VDSR', 'Set5', 2, 'SFSR_291plus.mat');
evalSetting(end+1) = evalSet('VDSR', 'VDSR', 'Set5', 3, 'SFSR_291plus.mat');
evalSetting(end+1) = evalSet('VDSR', 'VDSR', 'Set5', 4, 'SFSR_291plus.mat');

evalSetting(end+1) = evalSet('VDSR', 'VDSR', 'Set14', 2, 'SFSR_291plus.mat');
evalSetting(end+1) = evalSet('VDSR', 'VDSR', 'Set14', 3, 'SFSR_291plus.mat');
evalSetting(end+1) = evalSet('VDSR', 'VDSR', 'Set14', 4, 'SFSR_291plus.mat');

evalSetting(end+1) = evalSet('VDSR', 'VDSR', 'B100', 2, 'SFSR_291plus.mat');
evalSetting(end+1) = evalSet('VDSR', 'VDSR', 'B100', 3, 'SFSR_291plus.mat');
evalSetting(end+1) = evalSet('VDSR', 'VDSR', 'B100', 4, 'SFSR_291plus.mat');

evalSetting(end+1) = evalSet('VDSR', 'VDSR', 'Urban100', 2, 'SFSR_291plus.mat');
evalSetting(end+1) = evalSet('VDSR', 'VDSR', 'Urban100', 3, 'SFSR_291plus.mat');
evalSetting(end+1) = evalSet('VDSR', 'VDSR', 'Urban100', 4, 'SFSR_291plus.mat');

do.dataset = {'Set5','Set14','B100','Urban100'};
do.sf = [2 3 4];
depth = 5:20;
for i = 1:numel(do.dataset)
    for j = 1:numel(do.sf)
        for k = 1:numel(depth)
            evalSetting(end+1) = evalSet(['VDSRd',num2str(depth(k))], 'VDSR', do.dataset{i}, do.sf(j), ['exp_sf[2 3 4]_diff1_depth',num2str(depth(k)),'/best.mat']);
        end
    end
end
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
t2opts.exp = {'Bicubic','A+','RFL','SelfEx','SRCNN','RCN'};
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

g1opts.dataset = {'Set5'};%,'Set14','B100','Urban100'}; 
g1opts.sf = [3];
g1opts.exp = {'RCNd5','RCNd10','RCNd15','RCN'};
g1opts.graphSize = [];
g1opts.graphX = [5 10 15 20]-4;
g1opts.graphXname = {'1','6','11','16'};
g1opts.graphYname = [];
g1opts.graphName = 'graph1';
g1opts.graphDir = 'paper/figs';
g1opts.graphXlabel = 'Recursion';
g1opts.saved = [];
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
tiopts.exp = {'RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCN'};
indNormal = 1;
tiopts.interSetting = {[1 1 indNormal], [1 3 indNormal], [1 5 indNormal], [1 7 indNormal], [1 9 indNormal], [1 11 indNormal]};
tiopts.printTime = false;
tiopts.tableName = 'table_inter';
tiopts.fid = fileID;

%--------------------------------------------------------------------------

%%%% everything for paper

% texPrefix(fileID);
% % makeTable1(t1opts);
% t2opts.printTime = true;
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
% makeTableInter(tiopts);
% texSuffix(fileID);
% fclose(fileID);


%--------------------------------------------------------------------------
%%%% for Everything in paper DRCN
% 
% fileID3 = fopen('paper/redunt.tex','w');
% texPrefix(fileID3);
% % % 
% t2opts.fid = fileID3;
% t2opts.exp = {'Bicubic','A+','SRCNN','RFL','SelfEx','RCN'};
% makeTable2(t2opts);
% 
% f1opts.fid = fileID3;
% f1opts.boxSize = [30 50];
% f1opts.lineWidth = 10;
% f1opts.dataset = 'Urban100'; f1opts.imgNum = 39;
% makeFigure1(f1opts);
% f1opts.lineWidth = 5;
% f1opts.dataset = 'B100'; f1opts.imgNum = '220075';
% makeFigure1(f1opts);
% 
% % f1opts.exp = {'HR','SRCNN','SelfEx','RCN'};
% % f1opts.lineWidth = 15;
% % % makeFigure1(f1opts);
% % 
%  f2opts.fid = fileID3;
%  f2opts.figName = 'figDRCNred';
%  f2opts.figDir = 'paper/figs';
% % 
%  f2opts.sf = 2;
%  f2opts.boxSize = [30 30];
%  f2opts.lineWidth = 4;
%  f2opts.dataset = 'B100'; f2opts.imgNum = '58060';
%  makeFigure2(f2opts);
% 
% f2opts.sf = 4;
% f2opts.boxSize = [30 30];
% f2opts.lineWidth = 4;
% f2opts.dataset = 'B100'; f2opts.imgNum = '134035';
% makeFigure2(f2opts);
% 
% f2opts.sf = 3;
% f2opts.boxSize = [30 30];
% f2opts.lineWidth = 4;
% f2opts.dataset = 'Set14'; f2opts.imgNum = 'ppt3';
% makeFigure2(f2opts);
% 
% g1opts.fid = fileID3;
% makeGraph(g1opts);
% 
% tiopts.fid = fileID3;
% tiopts.exp = {'RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter', ...
%               'RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter', ...
%               'RCNInter','RCNInter','RCN'};
% indNormal = 4;
% tiopts.interSetting = {[1 1 indNormal], [2 2 indNormal], [3 3 indNormal], [4 4 indNormal],...
%                        [5 5 indNormal], [6 6 indNormal], [7 7 indNormal], [8 8 indNormal],...
%                        [9 9 indNormal], [10 10 indNormal], [11 11 indNormal], [12 12 indNormal],...
%                        [13 13 indNormal], [14 14 indNormal], [15 15 indNormal], [16 16 indNormal]};
% makeTableInter(tiopts);
% tiopts.exp = {'RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCN'};
% indNormal = 0;
% tiopts.interSetting = {[1 1 indNormal], [1 3 indNormal],  [1 5 indNormal],  [1 7 indNormal],...
%                        [1 9 indNormal], [1 11 indNormal], [1 13 indNormal], [1 15 indNormal]};
% makeTableInter(tiopts);
% tiopts.exp = {'RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCN'};
% indNormal = 2;
% tiopts.interSetting = {[1 1 indNormal], [1 3 indNormal],  [1 5 indNormal],  [1 7 indNormal],...
%                        [1 9 indNormal], [1 11 indNormal], [1 13 indNormal], [1 15 indNormal]};
% makeTableInter(tiopts);                   
% tiopts.exp = {'RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCN'};
% indNormal = 3;
% tiopts.interSetting = {[1 1 indNormal], [1 3 indNormal],  [1 5 indNormal],  [1 7 indNormal],...
%                        [1 9 indNormal], [1 11 indNormal], [1 13 indNormal], [1 15 indNormal]};
% makeTableInter(tiopts);
% tiopts.exp = {'RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCN'};
% indNormal = 4;
% tiopts.interSetting = {[1 1 indNormal], [1 3 indNormal],  [1 5 indNormal],  [1 7 indNormal],...
%                        [1 9 indNormal], [1 11 indNormal], [1 13 indNormal], [1 15 indNormal]};                   
% makeTableInter(tiopts);
% % 
% texSuffix(fileID3);
% fclose(fileID3);

%--------------------------------------------------------------------------
%%%% for supp. 

cp = 0;
SF = 3;

fileID2 = fopen(['paper/supplebook_DRCN_x',num2str(SF),'.tex'],'w');
texPrefix(fileID2);
fSuppopts.fid = fileID2;
fSuppopts.figName = ['figSupDRCNx',num2str(SF)'];
fSuppopts.sf = SF;
fSuppopts.exp = {'HR','Bicubic','A+','SRCNN','RFL','SelfEx','RCN'};
fSuppopts.figDir = 'paper/meetfig';
dataset = {'Set5','Set14','B100','Urban100'};
for d = 1:numel(dataset)
    fSuppopts.dataset = dataset{d};
    if strcmp(fSuppopts.dataset, 'Urban100')
        fSuppopts.lineWidth = 10;
        fSuppopts.boxSize = [60 60];
    else
        fSuppopts.lineWidth = 4;
        fSuppopts.boxSize = [30 30];
    end
    for i = 1:numel(dir(fullfile('data',fSuppopts.dataset)))-2
        fSuppopts.imgNum = i;
        makeFigureSupp(fSuppopts);
        cp = cp + 1;
        if cp == 6, clearpage(fileID2); cp = 0; end;
    end
end

texSuffix(fileID2);
fclose(fileID2);
%--------------------------------------------------------------------------
%%%% for supp VDSR
% cp = 0;
% SF = 2;
% 
% fileID2 = fopen(['paper/supplebook_VDSR_x',num2str(SF),'.tex'],'w');
% texPrefix(fileID2);
% fSuppopts.fid = fileID2;
% fSuppopts.figName = ['figSupVDSRx',num2str(SF)'];
% fSuppopts.sf = SF;
% fSuppopts.exp = {'HR','A+','SRCNN','RFL','SelfEx','VDSR'};
% fSuppopts.figDir = 'paper2/figs';
% dataset = {'Set5','Set14','B100','Urban100'};
% for d = 1:numel(dataset)
%     fSuppopts.dataset = dataset{d};
%     if strcmp(fSuppopts.dataset, 'Urban100')
%         fSuppopts.lineWidth = 10;
%         fSuppopts.boxSize = [60 60];
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
%%%% for paper 2, VDSR
%%%% everything for paper
% file2ID = fopen('paper2/forPaper2Fig.tex','w');
% texPrefix(file2ID);
% % 
% t2opts.fid = file2ID;
% t2opts.printTime = true;
% t2opts.exp = {'Bicubic','A+','RFL','SelfEx','SRCNN','VDSR'};
% makeTable2(t2opts);
% % 
% % f1opts.fid = file2ID;
% % f1opts.figDir = 'paper2/figs';
% % f1opts.boxSize = [40 60];
% % f1opts.lineWidth = 10;
% % f1opts.exp = {'HR','A+','RFL','SelfEx','SRCNN','VDSR'};
% % f1opts.dataset = 'Urban100'; f1opts.imgNum = 2;
% % makeFigure1(f1opts);
% % 
% f2opts.fid = file2ID;
% f2opts.figDir = 'paper2/figs';
% f2opts.figName = 'figVDSR';
% 
% f2opts.boxSize = [30 30];
% f2opts.lineWidth = 4;
% f2opts.exp = {'HR','A+','RFL','SelfEx','SRCNN','VDSR'};
% f2opts.dataset = 'B100'; f2opts.imgNum = '148026';
% makeFigure2(f2opts);
% 
% f2opts.boxSize = [30 30];
% f2opts.lineWidth = 4;
% f2opts.exp = {'HR','A+','RFL','SelfEx','SRCNN','VDSR'};
% f2opts.dataset = 'B100'; f2opts.imgNum = '38092';
% makeFigure2(f2opts);

% g1opts.fid = file2ID;
% g1opts.dataset = {'Set5','Set14','B100','Urban100'}; 
% g1opts.sf = [2 3 4];
% g1opts.exp = {'VDSRd5','VDSRd6','VDSRd7','VDSRd8','VDSRd9','VDSRd10','VDSRd11','VDSRd12','VDSRd13',...
%               'VDSRd14','VDSRd15','VDSRd16','VDSRd17','VDSRd18','VDSRd19','VDSRd20'};
% g1opts.graphSize = [];
% g1opts.graphX = [5:20];
% g1opts.graphXname = {'5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'};
% g1opts.graphYname = [];
% g1opts.graphXlabel = 'Depth';
% g1opts.graphName = 'graph1';
% g1opts.graphDir = 'paper2/figs';
% g1opts.printOne = false;
% g1opts.saved = '4data.mat';
% makeGraph(g1opts);
% texSuffix(file2ID);
% fclose(file2ID);
%--------------------------------------------------------------------------

% dataset ='Set5';
% imgNum = 3;
% boxSize = [40 40];
% boxPose = [30 30 ];
% lineWidth = 4;
% lineColor = [255 0 0];
% sf = 3;
% exp = {'RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCNInter','RCN'};
% indNormal = 4;
% interSetting = {[1 1 indNormal], [3 3 indNormal],  [5 5 indNormal],  [7 7 indNormal],...
%                 [9 9 indNormal], [11 11 indNormal], [13 13 indNormal], [15 15 indNormal]};
% figName = 'PROG';
% figDir = 'paper/figs';
% 
% % numColumn = numel(exp);
% 
% if ~exist(fullfile(figDir,figName),'dir')
%     mkdir(fullfile(figDir,figName));
% end
% datasetName = dataset;
% SF = sf;
% gtDir = fullfile('data',datasetName);
% outDir = fullfile('data','result');
% if isa(imgNum,'char')
%     imgNum = findImgNum(dataset, imgNum);
% end
% img_lst = dir(gtDir); img_lst = img_lst(3:end);
% [~,imgName,imgExt] = fileparts(img_lst(imgNum).name);
% imGT = imread(fullfile(gtDir, [imgName,imgExt]));
% imGT = modcrop(imGT, SF);
% 
% if isempty(boxSize)
%     boxSize(1) = ceil(size(imGT,1)/10);
%     boxSize(2) = ceil(size(imGT,2)/10);
% end
% 
% PSNR_array = zeros(numel(exp),1);
% SSIM_array = zeros(numel(exp),1);
% 
% for indExp = 1:numel(exp)
%     expName = exp{indExp};
%     if indExp == numel(exp)
%         outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
%     else
%         outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF),'_d',num2str(interSetting{indExp})]);
%     end
%     if ~strcmp(expName,'HR')
%         imSR = imread(fullfile(outDir, outRoute, [imgName,'.png']));
%         [psnr, ssim] = compute_diff(imGT, imSR, SF);
%         PSNR_array(indExp, 1) = psnr;
%         SSIM_array(indExp, 1) = ssim;
%         imSRcolor = colorize(imGT, imSR, SF);
%     else
%         if size(imGT,3) == 1, imGT = cat(3,imGT,imGT,imGT); end;
%         imSRcolor = modcrop(imGT, SF);
%     end
%     imSRcolor = shave(imSRcolor,[SF SF]); % for methods like A+, not predicting boundary.
%     ShapeInserter = vision.ShapeInserter('LineWidth',lineWidth,'BorderColor','Custom','CustomBorderColor',lineColor);
%     boximSRcolor = step(ShapeInserter, imSRcolor, int32(cat(2,fliplr(boxPose),fliplr(boxSize))));
%     subimSRcolor = imSRcolor(boxPose(1):boxPose(1)+boxSize(1)-1,boxPose(2):boxPose(2)+boxSize(2)-1,:);
%     subimSRcolor = imresize(subimSRcolor, [NaN,size(boximSRcolor,2)]);
% %     ShapeInserter = vision.ShapeInserter('LineWidth',lineWidth*SF,'BorderColor','Custom','CustomBorderColor',lineColor);
% %     subimSRcolor = step(ShapeInserter, subimSRcolor, int32([1,1,size(subimSRcolor,2),size(subimSRcolor,1)]));
% %     catimSRcolor = cat(1,boximSRcolor,subimSRcolor);
%     if indExp == numel(exp)
%         imwrite(subimSRcolor,fullfile(figDir,figName,[imgName,'_for_',figName,'_',expName,'.png']));
%     else
%         imwrite(subimSRcolor,fullfile(figDir,figName,[imgName,'_for_',figName,'_',expName,'_',num2str(interSetting{indExp}),'.png']));
% 
%     end
% end
% save(fullfile(figDir,figName,'PSNR.mat'), 'PSNR_array');

% % printTop(fSuppopts)
% Ours = 'VDSR';
% cp = 0;

% list = setList(2,'Set5',1);
% list(end+1) = setList(2,'B100','102061');
% list(end+1) = setList(2,'B100','145086');
% for u100list2 = [1 2 6 10 13 14 15 19 25 26 27 36 46 62 65 68 72 87 93 94 96]
%     list(end+1) = setList(2,'Urban100',u100list2);
% end
% 
% list(end+1) = setList(3,'B100','148026');
% list(end+1) = setList(3,'B100','219090');
% list(end+1) = setList(3,'B100','220075');
% for u100list3 = [2 25 52 68 75 82 87 96]
%     list(end+1) = setList(3,'Urban100',u100list3);
% end
% 
% list(end+1) = setList(4,'B100','134035');
% list(end+1) = setList(4,'B100','69015');
% list(end+1) = setList(4,'Urban100',2);
% list(end+1) = setList(4,'Urban100',40);
% list(end+1) = setList(4,'Urban100',92);
% 
% list = setList(2,'Set5',1);
% list(end+1) = setList(2,'B100','37073');
% for u100list2 = [2 19 25 26 39 46 72 75 83 96]
%     list(end+1) = setList(2,'Urban100',u100list2);
% end
% 
% list(end+1) = setList(3,'Set14','foreman');
% list(end+1) = setList(3,'Set14','ppt3');
% list(end+1) = setList(3,'B100','119082');
% list(end+1) = setList(3,'B100','219090');
% list(end+1) = setList(3,'B100','220075');
% list(end+1) = setList(3,'B100','69015');
% for u100list3 = [2 10 17 25 41 43 68]
%     list(end+1) = setList(3,'Urban100',u100list3);
% end
% list(end+1) = setList(4,'B100','106024');
% list(end+1) = setList(4,'B100','69015');
% list(end+1) = setList(4,'B100','86000');
% list(end+1) = setList(4,'Urban100',26);
% list(end+1) = setList(4,'Urban100',69);
% list(end+1) = setList(4,'Urban100',82);
% 
% 
% if strcmp(Ours,'RCN')
%     fileSup = fopen('paper/supplebook_DRCN_picked.tex','w');
%     fSuppopts.figName = 'figSupDRCNp';
%     fSuppopts.exp = {'HR','A+','SRCNN','RFL','SelfEx','RCN'};
%     fSuppopts.figDir = 'paper/figs';
% else
%     fileSup = fopen('paper2/supplebook_VDSR_picked.tex','w');
%     fSuppopts.figName = 'figSupVDSRp';
%     fSuppopts.exp = {'HR','A+','RFL','SelfEx','SRCNN','VDSR'};
%     fSuppopts.figDir = 'paper2/figs';
% end
% texPrefix(fileSup);
% fSuppopts.fid = fileSup;
% for i = 1:numel(list)
%     fSuppopts.sf = list(i).sf;
%     fSuppopts.dataset = list(i).dataset;
%     fSuppopts.imgNum = list(i).imgNum;
%     if strcmp(fSuppopts.dataset, 'Urban100')
%         fSuppopts.lineWidth = 10;
%         fSuppopts.boxSize = [60 60];
%     else
%         fSuppopts.lineWidth = 4;
%         fSuppopts.boxSize = [30 30];
%     end
%     makeFigureSupp(fSuppopts);
%     cp = cp + 1;
%     if cp == 6, clearpage(fileSup); cp = 0; end;
% end
% texSuffix(fileSup);
% fclose(fileSup);


function printTop(fSuppopts)

dataset = {'Set5', 'Set14', 'B100', 'Urban100'};
sf = [2 3 4];
Ours = 'VDSR';

gs = zeros(219*numel(sf),1);
is = zeros(219*numel(sf),1);
ds = cell(219*numel(sf),1);
ss = zeros(219*numel(sf),1);

indSave = 1;

if ~exist(['Top100for',Ours,'.mat'],'file')

    for indDataset = 1:numel(dataset)
        datasetName = dataset{indDataset};
        gtDir = fullfile('data',datasetName);
        outDir = fullfile('data','result');
        img_lst = dir(gtDir); img_lst = img_lst(3:end);
        numImg = numel(img_lst);    
        boxSize = [30 30];
        if strcmp(datasetName, 'Urban100')
            boxSize = [60 60];
        end

        for indSF = 1:numel(sf)
            SF = sf(indSF);
            imgNames = cell(numImg,1);

            for indImg = 1:numImg
                [~,imgNames{indImg},imgExt] = fileparts(img_lst(indImg).name);
                imGT = imread(fullfile(gtDir, [imgNames{indImg},imgExt]));  
                imGT = modcrop(imGT, SF);

                expName = 'SelfEx';
                outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
                imSR1 = imread(fullfile(outDir, outRoute, [imgNames{indImg},'.png']));

                expName = 'SRCNN';
                outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
                imSR2 = imread(fullfile(outDir, outRoute, [imgNames{indImg},'.png']));

                expName = 'A+';
                outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
                imSR3 = imread(fullfile(outDir, outRoute, [imgNames{indImg},'.png']));

                expName = Ours;
                outRoute = fullfile(expName, [expName,'_',datasetName,'_x',num2str(SF)]);
                imOurs = imread(fullfile(outDir, outRoute, [imgNames{indImg},'.png']));

                imGTs = shave(imGT, [SF SF]);
                imSR1s = shave(imSR1, [SF SF]);
                imSR2s = shave(imSR2, [SF SF]);
                imSR3s = shave(imSR3, [SF SF]);
                imOurss = shave(imOurs, [SF SF]);
                [~,~,maxVal] = findBestPos(imGTs, imSR2s, imSR3s, imSR1s, imOurss, boxSize);

                gs(indSave) = maxVal;
                is(indSave) = indImg;
                ds{indSave} = datasetName;
                ss(indSave) = SF;
                indSave = indSave + 1;
            end
        end
    end
    save(['Top100for',Ours,'.mat'],'gs','is','ds','ss');
else
    load(['Top100for',Ours,'.mat']);
end

cp = 0;

if strcmp(Ours,'RCN')
    fileID2 = fopen('paper/supplebook_DRCN_Top100.tex','w');
    fSuppopts.figName = 'figSupDRCNT';
    fSuppopts.exp = {'HR','A+','SRCNN','RFL','SelfEx','RCN'};
    fSuppopts.figDir = 'paper/figs';
else
    fileID2 = fopen('paper2/supplebook_VDSR_Top100.tex','w');
    fSuppopts.figName = 'figSupVDSRT';
    fSuppopts.exp = {'HR','A+','SRCNN','RFL','SelfEx','VDSR'};
    fSuppopts.figDir = 'paper2/figs';
end
texPrefix(fileID2);
fSuppopts.fid = fileID2;

[~,index] = sort(gs,'descend');
for i = 1:100
    fSuppopts.sf = ss(index(i));
    fSuppopts.dataset = ds{index(i)};
    fSuppopts.imgNum = is(index(i));
    if strcmp(fSuppopts.dataset, 'Urban100')
        fSuppopts.lineWidth = 10;
        fSuppopts.boxSize = [60 60];
    else
        fSuppopts.lineWidth = 4;
        fSuppopts.boxSize = [30 30];
    end
    makeFigureSupp(fSuppopts);
    cp = cp + 1;
    if cp == 6, clearpage(fileID2); cp = 0; end;
end
texSuffix(fileID2);
fclose(fileID2);

% function [maxX, maxY, max_val] = findBestPos(imGT, imSRCNN, imAplus, imSelfEx, imRCN, boxSize)
% % find the region that illustrates where SFFSR works well.
% % i.e. maximize PSNR(SFFSR Window ) - max (PSNR(A+ Window),
% % PSNR(SRCNNwindow))
% 
% if size(imGT,3) > 1
%     imGT = rgb2ycbcr(imGT);
%     imGT = imGT(:,:,1);
% end
% 
% if size(imSelfEx,3) > 1
%     imSelfEx = rgb2ycbcr(imSelfEx);
%     imSelfEx = imSelfEx(:,:,1);
% end
% 
% max_val = -1e5;
% maxX = 1;
% maxY = 1;
% sz = size(imGT);
% stride = 1;
% 
% [~, ssim1] = ssim(imRCN,   imGT, 'Exponents', [0 0 1]);
% [~, ssim2] = ssim(imSRCNN, imGT, 'Exponents', [0 0 1]);
% [~, ssim3] = ssim(imAplus, imGT, 'Exponents', [0 0 1]);
% [~, ssim4] = ssim(imSelfEx, imGT, 'Exponents', [0 0 1]);
% ssim1c = cumsum(cumsum(ssim1, 1), 2);
% ssim2c = cumsum(cumsum(ssim2, 1), 2);
% ssim3c = cumsum(cumsum(ssim3, 1), 2);
% ssim4c = cumsum(cumsum(ssim4, 1), 2);
% for x = 2:stride:sz(1)-boxSize(1)
%     for y=2:stride:sz(2)-boxSize(2)
%         val1 = ssim1c(x+boxSize(1)-1,y+boxSize(2)-1) - ssim1c(x+boxSize(1)-1,y-1) - ssim1c(x-1,y+boxSize(2)-1) + ssim1c(x-1,y-1);
%         val2 = ssim2c(x+boxSize(1)-1,y+boxSize(2)-1) - ssim2c(x+boxSize(1)-1,y-1) - ssim2c(x-1,y+boxSize(2)-1) + ssim2c(x-1,y-1);
%         val3 = ssim3c(x+boxSize(1)-1,y+boxSize(2)-1) - ssim3c(x+boxSize(1)-1,y-1) - ssim3c(x-1,y+boxSize(2)-1) + ssim3c(x-1,y-1);
%         val4 = ssim4c(x+boxSize(1)-1,y+boxSize(2)-1) - ssim4c(x+boxSize(1)-1,y-1) - ssim4c(x-1,y+boxSize(2)-1) + ssim4c(x-1,y-1);
%         if log(val1)-log(max(val2,max(val3,val4))) > max_val
%             max_val= log(val1)-log(max(val2,max(val3,val4)));
%             maxX = x;
%             maxY = y;
%         end
%     end
% end

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
if numel(exp) >= 7
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
    elseif strcmp('VDSR', exp{indExp})
        fprintf(fid,[' & VDSR (Ours)']);
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
        tavgTIME = avgTIME; tavgTIME(1) = 100;
        [~,maxAvgPSNR] = max(avgPSNR,[],2); [~,maxAvgSSIM] = max(avgSSIM,[],2);
        [~,secmaxAvgPSNR] = secmax(avgPSNR,2); [~,secmaxAvgSSIM] = secmax(avgSSIM,2); 
        [~,minAvgTIME] = max(-1*tavgTIME,[],2); [~,secminAvgTIME] = secmax(-1*tavgTIME,2);
                
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
                    if indExp == minAvgTIME
                        fprintf(fid, ['/{\\color{red}',num2str(avgTIME(1, indExp),'%.2f'),'}']);
                    elseif indExp == secminAvgTIME
                        fprintf(fid, ['/{\\color{blue}',num2str(avgTIME(1, indExp),'%.2f'),'}']);
                    else
                        fprintf(fid, ['/',num2str(avgTIME(1, indExp),'%.2f')]);
                    end                        
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
                    if indExp == minAvgTIME
                        fprintf(fid, ['/{\\color{red}',num2str(avgTIME(1, indExp),'%.2f'),'}']);
                    elseif indExp == secminAvgTIME
                        fprintf(fid, ['/{\\color{blue}',num2str(avgTIME(1, indExp),'%.2f'),'}']);
                    else
                        fprintf(fid, ['/',num2str(avgTIME(1, indExp),'%.2f')]);
                    end                        
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
                    if indExp == minAvgTIME
                        fprintf(fid, ['/{\\color{red}',num2str(avgTIME(1, indExp),'%.2f'),'}']);
                    elseif indExp == secminAvgTIME
                        fprintf(fid, ['/{\\color{blue}',num2str(avgTIME(1, indExp),'%.2f'),'}']);
                    else
                        fprintf(fid, ['/',num2str(avgTIME(1, indExp),'%.2f')]);
                    end                        
                end
            end
        end
        fprintf(fid, '\\\\\n');        
    end
    fprintf(fid,'\\hline\n');
end
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,'\\caption{Benchmark results. Average PSNR/SSIM for scale factor ');
for i=1:numel(sf)
    if i < numel(sf)-1
        fprintf(fid,['$\\times$',num2str(sf(i)),', ']);
    elseif i == numel(sf)-1
        fprintf(fid,['$\\times$',num2str(sf(i)),' and ']);
    else
        fprintf(fid,['$\\times$',num2str(sf(i))]); 
    end
end
fprintf(fid,' on datasets \\textit{');
for i=1:numel(dataset)    
    if i < numel(dataset)-1
        fprintf(fid,[dataset{i}, ', ']);
    elseif i == numel(dataset)-1
        fprintf(fid,[dataset{i}, ' and ']);
    else
        fprintf(fid,dataset{i});
    end
end
fprintf(fid,'}. {\\color{red}Red color} indicates the best performance and {\\color{blue}blue color} indicates the second best one.}\n');
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
if isa(imgNum,'char')
    imgNum = findImgNum(dataset, imgNum);
end
img_lst = dir(gtDir); img_lst = img_lst(3:end);
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
    subimSRcolor = imresize(subimSRcolor, [NaN, size(imGT,2)]);
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
if size(imGTbox,1) > size(imGTbox,2)
    GTwidthVal = 0.15;
else
    GTwidthVal = 0.25;
end
fprintf(fid, ['\\multirow{4}{*}{\\graphicspath{{figs/',figName,'/}}\\includegraphics[width=',num2str(GTwidthVal),'\\textwidth]{', ...
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
                fprintf(fid, ['& Ground Truth (PSNR, SSIM)']);
            else
                if strcmp(exp{indExp2},'RCN')
                    fprintf(fid, ['& DRCN (Ours) (']);
                elseif strcmp(exp{indExp2},'VDSR')
                    fprintf(fid, ['& VDSR (Ours) (']);
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
fprintf(fid,['\\caption{Super-resolution results of ``',setValidName(imgName,'_'),'"(\\textit{',datasetName,'}) with scale factor $\\times$',num2str(SF),'. Our result is visually pleasing.}\n']);
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
if isa(imgNum,'char')
    imgNum = findImgNum(dataset, imgNum);
end
img_lst = dir(gtDir); img_lst = img_lst(3:end);
[~,imgName,imgExt] = fileparts(img_lst(imgNum).name);
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
        fprintf(fid, 'Ground Truth');
    elseif strcmp(exp{indColumn},'RCN')
        fprintf(fid, ['& DRCN (Ours)']);
    elseif strcmp(exp{indColumn},'VDSR')
        fprintf(fid, ['& VDSR (Ours) ']);
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
fprintf(fid,['\\caption{Super-resolution results of ``',setValidName(imgName,'_'),'"(\\textit{',datasetName,'}) with scale factor $\\times$',num2str(SF),'. Our result is visually pleasing.}\n']);
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

if strcmp(exp{numel(exp)},'VDSR')
    vdsrflag = 1;
else
    vdsrflag  =0;
end

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
[~,imgName,imgExt] = fileparts(img_lst(imgNum).name);
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
%         elseif strcmp(expName,'SelfEx')
%             imSelfEx = imread(fullfile(outDir, outRoute, [imgName,'.png']));
        elseif indExp == numel(exp)
            imRCN = imread(fullfile(outDir, outRoute, [imgName,'.png']));
        end
    end
    imGTs = shave(imGT,[SF SF]);
    imSRCNN = shave(imSRCNN,[SF SF]);
    imAplus = shave(imAplus,[SF SF]);
%     imSelfEx = shave(imSelfEx, [SF SF]);
    imRCN = shave(imRCN,[SF SF]);
%     [boxPose(1),boxPose(2),~] = findBestPos(imGTs, imSRCNN, imAplus, imSelfEx, imRCN, boxSize);
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
    imwrite(catimSRcolor,fullfile(figDir,figName,[imgName,'_for_',figName,'_',expName,'_x',num2str(SF),'.png']));
end

[~,maxPSNR] = max(PSNR_array,[],1); [~,maxSSIM] = max(SSIM_array,[],1);
[~,secmaxPSNR] = secmax(PSNR_array,1); [~,secmaxSSIM] = secmax(SSIM_array,1); 

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
                             [imgName,'_for_',figName,'_',exp{indExp},'_x',num2str(SF),'.png'],'}}\n']);

            else
                fprintf(fid, ['& {\\graphicspath{{figs/',figName,'/}}\\includegraphics[height=',num2str(0.85/2,'%.2f'),'\\textheight]{', ...
                             [imgName,'_for_',figName,'_',exp{indExp},'_x',num2str(SF),'.png'],'}}\n']);       
            end
        else
            if indColumn == 1
                fprintf(fid, ['{\\graphicspath{{figs/',figName,'/}}\\includegraphics[width=',num2str(0.85/numColumn,'%.2f'),'\\textwidth]{', ...
                             [imgName,'_for_',figName,'_',exp{indExp},'_x',num2str(SF),'.png'],'}}\n']);
            else
                fprintf(fid, ['& {\\graphicspath{{figs/',figName,'/}}\\includegraphics[width=',num2str(0.85/numColumn,'%.2f'),'\\textwidth]{', ...
                             [imgName,'_for_',figName,'_',exp{indExp},'_x',num2str(SF),'.png'],'}}\n']);       
            end     
        end
    end    
    fprintf(fid, '\\\\\n');
    for indColumn = 1:numColumn
        indExp2 = indExp2 + 1;
        if indColumn ~= 1
            fprintf(fid, '& ');
        end
        if strcmp(exp{indExp2},'HR')
            fprintf(fid, 'Ground Truth');
        elseif strcmp(exp{indExp2},'RCN')
            fprintf(fid, 'DRCN (Ours)');
        elseif strcmp(exp{indExp2},'VDSR')
            fprintf(fid, 'VDSR (Ours)');
        elseif strcmp(exp{indExp2},'A+')
            fprintf(fid, [exp{indExp2}, ' \\cite{Timofte}']);
        elseif strcmp(exp{indExp2},'RFL')
            fprintf(fid, [exp{indExp2}, ' \\cite{schulter2015fast}']);
        elseif strcmp(exp{indExp2},'SelfEx')
            fprintf(fid, [exp{indExp2}, ' \\cite{Huang-CVPR-2015}']);            
        elseif strcmp(exp{indExp2},'SRCNN') && vdsrflag == 0
            fprintf(fid, [exp{indExp2}, ' \\cite{dong2014image}']);
        elseif strcmp(exp{indExp2},'SRCNN') && vdsrflag == 1
            fprintf(fid, [exp{indExp2}, ' \\cite{dong2014image}']);
        end
    end
    fprintf(fid, '\\\\\n');
    for indColumn = 1:numColumn
        indExp3 = indExp3 + 1;
        if indColumn ~= 1 
            fprintf(fid, '& ');
        end
        if strcmp(exp{indExp3},'HR')
            fprintf(fid, '(PSNR, SSIM)');
        else
            if indExp3 == maxPSNR
                fprintf(fid,['({\\color{red}{',num2str(PSNR_array(indExp3),'%.2f'),'}}, ']);
                if indExp3 == maxSSIM
                    fprintf(fid,['{\\color{red}{',num2str(SSIM_array(indExp3),'%.4f'),'}})']);
                elseif indExp3 == secmaxSSIM
                    fprintf(fid,['{\\color{blue}{',num2str(SSIM_array(indExp3),'%.4f'),'}})']);
                else
                    fprintf(fid,[num2str(SSIM_array(indExp3),'%.4f'),')']);
                end
            elseif indExp3 == secmaxPSNR
                fprintf(fid,['({\\color{blue}{',num2str(PSNR_array(indExp3),'%.2f'),'}}, ']);
                if indExp3 == maxSSIM
                    fprintf(fid,['{\\color{red}{',num2str(SSIM_array(indExp3),'%.4f'),'}})']);
                elseif indExp3 == secmaxSSIM
                    fprintf(fid,['{\\color{blue}{',num2str(SSIM_array(indExp3),'%.4f'),'}})']);
                else
                    fprintf(fid,[num2str(SSIM_array(indExp3),'%.4f'),')']);
                end
            else
                fprintf(fid,['(',num2str(PSNR_array(indExp3),'%.2f'),', ']);
                if indExp3 == maxSSIM
                    fprintf(fid,['{\\color{red}{',num2str(SSIM_array(indExp3),'%.4f'),'}})']);
                elseif indExp3 == secmaxSSIM
                    fprintf(fid,['{\\color{blue}{',num2str(SSIM_array(indExp3),'%.4f'),'}})']);
                else
                    fprintf(fid,[num2str(SSIM_array(indExp3),'%.4f'),')']);
                end
            end
        end
    end
    fprintf(fid, '\\\\\n');
end
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,['\\caption{Super-resolution results of ``',setValidName(imgName,'_'),'" (\\textit{',datasetName,'}) with scale factor $\\times$',num2str(SF),'.}\n']);
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
    fprintf(fid,'\\normalsize\n');
else
    fprintf(fid,'\\normalsize\n');
end
fprintf(fid,'\\begin{tabular}{ |');
for indColumn = 1:numel(exp)+1
    fprintf(fid,' c |');
end
fprintf(fid,' }\n\\hline\n');
fprintf(fid,'D');
for indExp = 1:numel(exp)
    if strcmp('RCN', exp{indExp})
        fprintf(fid,[' & Ensemble']);
    else
        if numel(interSetting{indExp}) > 1
            %fprintf(fid,[' & $\\sum\\limits_{d=1}^{',num2str(interSetting{indExp}(2)),'}$ Output $d$']);
            fprintf(fid, [' & ',num2str(interSetting{indExp}(2))]);
        else
            fprintf(fid,[' & Output ',num2str(interSetting{indExp}(2))]);
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
% fprintf(fid,['\\caption{Experiment on the effect of ensemble.',...
%     ' Quantitative evaluation (PSNR) on dataset Set5 is provided for scale factors $\\times2$,$\\times3$ and $\\times4$.',...
%     ' As more outputs are ensembled, the result becomes better.']);
fprintf(fid, ['\\caption{']);
    if interSetting{1}(3) == 1
        fprintf(fid, ' Sum of weight are normalized.');       
    elseif interSetting{1}(3) ==2
        fprintf(fid, ' Sum of weight are one.');   
    elseif interSetting{1}(3) ==3
        fprintf(fid, ' Averaged.');
    elseif interSetting{1}(3) ==0
        fprintf(fid, ' Use same weight with Ensemble');
    elseif interSetting{1}(3) ==4
        fprintf(fid, ' Just Add');
    end
% fprintf(fid,' {\\color{red}Red color} indicates the best performance.}\n');
fprintf(fid, '}\n');
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
Xlabel = opts.graphXlabel;
saved = opts.saved;

if ~exist(fullfile(graphDir,graphName),'dir')
    mkdir(fullfile(graphDir,graphName));
end

if isempty(graphSize)
    graphSize = [400 400];
end

PSNR_table = cell(numel(dataset),numel(sf));
SSIM_table = cell(numel(dataset),numel(sf));

if exist(saved,'file')
    load(saved);
else
    for indDataset = 1:numel(dataset)
        datasetName = dataset{indDataset};
        gtDir = fullfile('data',datasetName);
        outDir = fullfile('data','result');
        img_lst = dir(gtDir); img_lst = img_lst(3:end);
        numImg = numel(img_lst);

        for indSF = 1:numel(sf)
            SF = sf(indSF);
            PSNR_table{indDataset,indSF} = zeros(numImg, numel(exp));
            SSIM_table{indDataset,indSF} = zeros(numImg, numel(exp));
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
    if ~isempty(saved)
        save(saved,'PSNR_table','SSIM_table');
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
            if numel(sf)~=1
                legendInfo{indDataset} = [dataset{indDataset},' \times',num2str(sf(indSF))];
            else
                legendInfo{indDataset} = [dataset{indDataset}];
            end
        end
    end
    if numel(sf)~=1
        legend(legendInfo, 'Location', 'best', 'FontSize', 13); 
    end
    tightfig;
    print(fullfile(graphDir,graphName,'graphOne'), '-dpdf', '-r600');
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
            ax.FontSize = 13;
            ax.LineWidth = lw;
            hold on;
            %ax.FontWeight = 'bold';
            xlabel(Xlabel);
            ylabel('PSNR (dB)');
            grid on;
            set(gcf, 'PaperPosition', [0, 0, 5, 5]);
%             if numel(sf)~=1
%                 legendInfo{indDataset} = [dataset{indDataset},' \times',num2str(sf(indSF))];
%             else
                legendInfo{indDataset} = [dataset{indDataset}];
%             end
        end
        legend(legendInfo, 'Location', 'west', 'FontSize', 10);
        tightfig;
        print(fullfile(graphDir,graphName,['graph',num2str(sf(indSF))]), '-dpdf', '-r600');
    end
end

if printOne
    fprintf(fid,'\\begin{figure}\n');
    fprintf(fid,'\\begin{adjustwidth}{0cm}{-0.0cm}\n');
    fprintf(fid,'\\centering\n');
    fprintf(fid,['{\\graphicspath{{figs/',graphName,'/}}\\includegraphics[height=5.0cm]{graphOne.pdf}}\n']);
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
        fprintf(fid,[' on the dataset \\textit{']);
    else
        fprintf(fid,[' on datasets \\textit{']);
    end
    for i=1:numel(dataset)
        if i < numel(dataset)-1
            fprintf(fid,[dataset{i},', ']);
        elseif i == numel(dataset)-1
            fprintf(fid,[dataset{i},', ']);
        else
            fprintf(fid,[dataset{i},'}.\n']);
        end
    end
    fprintf(fid,'More recursions yielding larger receptive fields lead to better performances. (Graph {\\color{red}not complete} yet)}\n');
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

function lst = setList(sf, dataset, imgNum)
lst.sf = sf;
lst.dataset = dataset;
lst.imgNum = imgNum;

        
