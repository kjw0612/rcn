function [net, info] = rcn_dag(varargin)

% Image Restoration Network
% Author: Jiwon Kim (jiwon@alum.mit.edu), Jonghyuk
run(fullfile(fileparts(mfilename('fullpath')),...
  'snudeep', 'matlab', 'vl_setupnn.m')) ;

%% Download data
if ~exist('data', 'dir'), mkdir('data'); end
if ~exist('data/91', 'dir')
    url = 'https://www.dropbox.com/s/sngf409t615mq9c/sr_data_91_291.zip?dl=1';
    fprintf('Downloading images (91 and BSDS200) : %s\n', url);
    unzip(url, 'data');
    fprintf('Images Prepared. Two folders 91 and 291 (91+BSDS200)\n');
end
if ~exist('data/Set5', 'dir')
    url = 'https://www.dropbox.com/s/v7ffhhoodo0xstv/Set5.zip?dl=1';
    fprintf('Downloading images (Set5) : %s\n', url);
    unzip(url, 'data');
    fprintf('Images Prepared. Set5\n');
end


%% Set Options
opts.problems = {struct('type', 'SR', 'sf', 3)};
%opts.problems = {struct('type', 'SR', 'sf', 3), struct('type', 'JPEG', 'q', 20), struct('type', 'DENOISE', 'v', 0.001)};
opts.gpus = 2;
opts.resid = 1;
opts.recursive = 1;
opts.dropout = 1;
opts.depth = 10; % 10 optimal5
opts.filterSize = 64;
if opts.dropout, opts.filterSize = opts.filterSize * 8; end
opts.pad = 0;
opts.useBnorm = false;
exp_name = 'exp';
if opts.useBnorm
    exp_name = 'exp_bn';
end
for problem_iter = 1:numel(opts.problems)
    problem = opts.problems{problem_iter};
    switch problem.type
        case 'SR'
            exp_name = strcat(exp_name, '_S', num2str(problem.sf));
        case 'JPEG'
            exp_name = strcat(exp_name, '_J', num2str(problem.q));
        case 'DENOISE'
            exp_name = strcat(exp_name, '_N', num2str(problem.v));
    end
end
exp_name = sprintf('multi_obj_%s_resid%d_depth%d', exp_name, opts.resid, opts.depth);
opts.expDir = fullfile('data','exp',exp_name);
opts.dataDir = fullfile('data', '91');
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.train.batchSize = 64;
rep = 20;
if opts.dropout, rep = rep * 5; end
opts.train.learningRate = [0.1*ones(1,rep) 0.01*ones(1,rep) 0.001*ones(1,rep) 0.0001*ones(1,rep)];%*0.99 .^ (0:500);
opts.train.numEpochs = numel(opts.train.learningRate);
opts.train.continue = 0;
opts.train.gradRange = 1e-4;
opts.train.sync = true;
opts.train.expDir = opts.expDir;
opts.train.gpus = opts.gpus;
opts.train.numSubBatches = 1 ;
opts.train.testPath = fullfile('data', 'Set5', 'baby_GT.bmp');
opts.train.dropout = opts.dropout;
opts.train.recursive = opts.recursive;

opts = vl_argparse(opts, varargin);

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file') & 0
  imdb = load(opts.imdbPath) ;
else
  imdb = getRcnImdb(opts.dataDir, opts.problems, opts.depth, opts.pad, opts.resid);
  mkdir(opts.expDir) ;
  %save(opts.imdbPath, '-struct', 'imdb') ;
end

[net, opts.train.derOutputs] = rcn_init_dag(opts);
net.initParams();
%net = dagnn.DagNN.fromSimpleNN(net) ;
%net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
%             {'prediction','label'}, 'error') ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

info = rcn_train_dag(net, imdb, @getBatch, ...
                     opts.train, ...
                     'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------
if numel(opts.gpus) > 0
    inputs = {'input', gpuArray(imdb.images.data(:,:,:,batch)), ...
          'label', gpuArray(imdb.images.labels(:,:,:,batch))} ;
else
    inputs = {'input', imdb.images.data(:,:,:,batch), ...
          'label', imdb.images.labels(:,:,:,batch)} ;
end

function imdb = getRcnImdb(dataDir, problems, depth, pad, diff)
f_lst = dir(fullfile(dataDir, '*.*'));
ps = 2*depth+1; % patch size
stride = ps;%31;%ps - 2*pad;

nPatches = 0;
for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
    if f_info.isdir, continue; end
    im = imread(fullfile(dataDir, f_info.name));
    im = rgb2ycbcr(im);
    im = im(:,:,1);
    
    for problem_iter = 1:numel(problems)
        problem = problems{problem_iter};
        switch problem.type
            case 'SR'
                imhigh = modcrop(im, problem.sf);
            case 'JPEG'
                imhigh = im;
            case 'DENOISE'
                imhigh = im;
        end
        for j = 1:stride:size(imhigh,1)-ps+1
            for k = 1:stride:size(imhigh,2)-ps+1
                nPatches = nPatches +1;
            end
        end
    end
end

imsuba = zeros(nPatches, ps-pad*2, ps-pad*2);
imsublowa = zeros(nPatches, ps, ps);
ind = 0;
for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
    if f_info.isdir, continue; end
    im = imread(fullfile(dataDir, f_info.name));
    im = rgb2ycbcr(im);
    im = im(:,:,1);
    
    for problem_iter = 1:numel(problems)
        problem = problems{problem_iter};
        switch problem.type
            case 'SR'
                imhigh = modcrop(im, problem.sf);
                imhigh = single(imhigh)/255;
                imlow = imresize(imhigh, 1/problem.sf, 'bicubic');
                imlow = imresize(imlow, size(imhigh), 'bicubic');
                imlow = max(16.0/255, min(235.0/255, imlow));
            case 'JPEG'
                imhigh = single(im)/255;
                imwrite(imhigh, 'data/_temp.jpg', 'Quality', problem.q);
                imlow = imread('data/_temp.jpg');
                imlow = single(imlow)/255;
                delete('data/_temp.jpg');
            case 'DENOISE'
                imhigh = single(im)/255;
                imlow = single(imnoise(imhigh, 'gaussian', 0, problem.v));
        end
        if diff, imhigh=imhigh-imlow; end;
        for j = 1:stride:size(imhigh,1)-ps+1
            for k = 1:stride:size(imhigh,2)-ps+1
                imsub = imhigh(j+pad:j+ps-1-pad,k+pad:k+ps-1-pad);
                imsublow = imlow(j:j+ps-1,k:k+ps-1);
                ind = ind + 1;
                imsuba(ind,:,:,:)=imsub;
                imsublowa(ind,:,:,:)=imsublow;
            end
        end
    end
end

fprintf('Total Subimages %d\n', ind);
imsuba = imsuba(1:ind,:,:);
imsublowa = imsublowa(1:ind,:,:);

s = randperm(ind); %shuffle
imsuba = imsuba(s,:,:);
imsublowa = imsublowa(s,:,:);

imsuba = reshape(imsuba, ind, ps-2*pad, ps-2*pad, 1);
imsublowa = reshape(imsublowa, ind, ps, ps, 1);
imsuba = permute(imsuba, [2 3 4 1]);
imsublowa = permute(imsublowa, [2 3 4 1]);
imdb.images.data = single(imsublowa);
imdb.images.labels = single(imsuba);
val_size = floor(ind * 0.01);
imdb.images.set = [ones(1, ind-val_size) 3*ones(1, val_size)];
imdb.meta.sets = {'train', 'val', 'test'} ;
