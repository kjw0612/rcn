function [net, info] = rcn(varargin)
% Image Restoration Network
% Author: Jiwon Kim (jiwon@alum.mit.edu), Jonghyuk
run(fullfile(fileparts(mfilename('fullpath')),...
  'snudeep', 'matlab', 'vl_setupnn.m')) ;

%% Prepare data
if ~exist('data', 'dir'), mkdir('data'); end
if ~exist('data/91', 'dir')
    url = 'https://www.dropbox.com/s/sngf409t615mq9c/sr_data_91_291.zip?dl=1';
    fprintf('Downloading images : %s\n', url);
    unzip(url, 'data');
    fprintf('Images Prepared. Two folders 91 and 291 (91+BSDS200)\n');
end

%% Set Options
%opts.problems = {struct('type', 'SR', 'sf', 3)};
%opts.problems = {struct('type', 'JPEG', 'q', 20)};
%opts.problems = {struct('type', 'DENOISE', 'v', 0.001)};
opts.problems = {struct('type', 'SR', 'sf', 3), struct('type', 'JPEG', 'q', 20), struct('type', 'DENOISE', 'v', 0.001)};
opts.resid = 1;
opts.depth = 10; % 10 optimal
opts.continue =0;
opts.gradRange = 1e-4;
opts.sync = true;
opts.dataDir = 'data/91'; % 'data/291' for bsds 
opts.filterSize = 64;% number of filters
opts.useBnorm = true;
opts = vl_argparse(opts, varargin);

exp_name = 'exp';
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
exp_name = sprintf('%s_resid%d_depth%d', exp_name, opts.resid, opts.depth);
opts.expDir = fullfile('data/exp',exp_name);

rep=20;
opts.learningRate = [0.1*ones(1,rep) 0.01*ones(1,rep) 0.001*ones(1,rep) 0.0001*ones(1,rep)];%*0.99 .^ (0:500);
opts.gradRange = 1e-4;
if ~exist('data/result', 'dir'), mkdir('data/result'); end
opts.fname = sprintf('data/result/%s.txt',exp_name);
if opts.depth <= 20
    opts.batchSize = 64;
else
    opts.batchSize = 32;
end;

opts.weightDecay = 0.0001;

opts.numEpochs = numel(opts.learningRate);
opts.pad = 0;
opts.plotDiagnostics = 0 ;

if ~exist('data/trainData', 'dir'), mkdir('data/trainData'); end
if ~opts.continue || ~exist(sprintf('data/trainData/%s.mat',exp_name),'file')
    imdb = generateData(opts.dataDir, opts.problems, opts.depth, opts.pad, opts.resid);
    save(sprintf('data/trainData/%s.mat',exp_name), 'imdb');
else
    load(sprintf('data/trainData/%s.mat',exp_name));
end

% define net
net = rcn_init(opts);

opts = rmfield(opts, {'depth', 'filterSize', 'dataDir'}); % remove irrelavant fields
[net, info] = rcn_train(net, imdb, @getBatch, opts) ;

function imdb = generateData(dataDir, problems, depth, pad, diff)
f_lst = dir(fullfile(dataDir, '*.*'));
ps = (2*depth+1); % patch size
stride = ps;%ps - 2*pad;

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
imdb.images.set = [ones(1, ind-val_size) 2*ones(1, val_size)];
% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(:,:,:,batch) ;