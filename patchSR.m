clear;
p = pwd;
addpath(genpath(fullfile(p, 'methods')));  % the upscaling methods
addpath(fullfile(p, 'utils'));  % utils
addpath(genpath(fullfile(p, 'toolbox')));
run('snudeep/matlab/vl_setupnn.m');

model = 'sf3/best_D15_F256.mat';
modelPath = ['methods/RCN/',model];
gpu = 1;
SF = 3;
datasetName = 'Set5';
managableMax = 630*630;

load(modelPath);
net = dagnn.DagNN.loadobj(net) ;
if gpu
    net.move('gpu');
end
dataDir = fullfile('data', datasetName);
f_lst = dir(fullfile(dataDir, '*.*'));

numRepeat = 5;
timetable = zeros(numRepeat,1);

for indRepeat = 1:numRepeat
    for f_iter = 4%:numel(f_lst)
        f_info = f_lst(f_iter);
        if f_info.isdir, continue; end
        [~,imgName,~] = fileparts(f_lst(f_iter).name);
        im = imread(fullfile(dataDir, f_info.name));
        if size(im,3)>1
            im = rgb2ycbcr(im);
            im = im(:,:,1);
        end
        imhigh = modcrop(im, SF);
        imhigh = single(imhigh)/255;
        imlow = imresize(imhigh, 1/SF, 'bicubic');
        imlow = imresize(imlow, size(imhigh), 'bicubic');
        imlow = max(16.0/255, min(235.0/255, imlow));    
        if size(imlow,1)*size(imlow,2) > managableMax
    %         fprintf(['too big ', num2str(countBig), ' name : ', imgName,'\n']);
            impred1 = runPatchRCN(net, imlow, gpu, 15);
        else
            fprintf(['name : ',imgName, ', size : ',num2str([size(imlow,1), size(imlow,2)]),'\n']);
            tic;            
            if gpu, imlow = gpuArray(imlow); end;            
            impred2 = runRCN(net, imlow, gpu);
            timetable(indRepeat,1) = toc;
        end
    end
end
