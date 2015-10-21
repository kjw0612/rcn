function [net, info] = rcn_dag(varargin)
% Recursive Convolutional Network for Image Super-Resolution (CVPR 2016 Submission)
% Authors: Jiwon Kim (jiwon@alum.mit.edu), Jung Kwon Lee, Jonghyuk Lee
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
opts.gpus = 2;
minimal = 1;

opts.test_sf = 3;
opts.resid = 1;
opts.recursive = 1;
opts.deep_supervise = 1;
if minimal
  opts.dropout = 0;
  opts.depth = 10; % 10 optimal5
  opts.filterSize = 64; % 256 for depth 20 
  opts.augment = false; %data augmentation
else
  opts.dropout = 1;
  opts.depth = 10; % 10 optimal5
  opts.filterSize = 256; % 256 for depth 20 
  opts.augment = true; %data augmentation
end

opts.pad = 0;
opts.momentum = 0.9;
opts.useBnorm = false;
exp_name = 'exp';
if opts.useBnorm
    exp_name = 'exp_bn';
end
opts = vl_argparse(opts, varargin);
exp_name = sprintf('multi_obj_%s_depth%d', exp_name,  opts.depth);
opts.expDir = fullfile('data','exp',exp_name);
opts.dataDir = fullfile('data', '91');
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

if opts.depth <= 20
  opts.train.batchSize = 64;
elseif opts.depth <= 30
  opts.train.batchSize = 32; % 18
else
  opts.train.batchSize = 16;    % 4 
end;
rep = 20*3;
if opts.dropout, rep = rep * 5; opts.train.learningRate = [0.1*ones(1,rep) 0.01*ones(1,rep) 0.001*ones(1,rep) 0.0001*ones(1,rep)]; end %*0.99 .^ (0:500); 
opts.train.learningRate = 0.1; %[0.1*ones(1,rep) 0.01*ones(1,rep) 0.001*ones(1,rep) 0.0001*ones(1,rep)];%*0.99 .^ (0:500);
opts.train.numEpochs = numel(opts.train.learningRate);
opts.train.continue = 0;
if opts.depth <= 10
  opts.train.gradRange = 1e-4;
elseif opts.depth <= 20
  opts.train.gradRange = 0.1*1e-4;
else 
  opts.train.gradRange = 0.01*1e-4;
end
opts.train.useBnorm = opts.useBnorm;
opts.train.sync = true;
opts.train.expDir = opts.expDir;
opts.train.gpus = opts.gpus;
opts.train.dropout = opts.dropout;
opts.train.recursive = opts.recursive;
opts.train.momentum = opts.momentum;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file') & 0
  imdb = load(opts.imdbPath) ;
else
  imdb = getRcnImdb(opts.dataDir, opts.test_sf, opts.depth, opts.pad, opts.resid, opts.augment);
  mkdir(opts.expDir) ;
  %save(opts.imdbPath, '-struct', 'imdb') ;
end

% --------------------------------------------------------------------
%                                                         Initialization
% Look for a network for initialization
% If both depth and filter sizses exactly match, it's the best.
% Otherwise, search a network with the closest depth and the same number
% of filters.
% --------------------------------------------------------------------

[net, opts.train.derOutputs] = rcn_init_dag(opts);
net.initParams();
if ~minimal
  close_depth = 0;
  for i=1:100
    bestPath = sprintf('best/best_D%d_F%d.mat', i, opts.filterSize);
    if exist(bestPath, 'file') && abs(i - opts.depth) < abs(close_depth - opts.depth)
      load(bestPath);
      best_net = net;
      close_depth = i;
    end
  end
  if close_depth > 0, net.params.value = best_net.params.value; end
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

info = rcn_train_dag(net, imdb, @getBatch, ...
                     opts.depth, opts.filterSize, opts.train, ...
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

function imdb = getRcnImdb(dataDir, test_sf, depth, pad, diff, augment)
f_lst = dir(fullfile(dataDir, '*.*'));
ps = 2*depth+1; % patch size
stride = 21;%ps;%31;%ps - 2*pad;

nPatches = 0;
for sf = test_sf
  for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
    if f_info.isdir, continue; end
    im = imread(fullfile(dataDir, f_info.name));
    im = rgb2ycbcr(im);
    im = im(:,:,1);
    
    imhigh = modcrop(im, sf);
    
    for j = 1:stride:size(imhigh,1)-ps+1
      for k = 1:stride:size(imhigh,2)-ps+1
        if augment
         nPatches = nPatches +16;
        else
         nPatches = nPatches +1;
        end
      end
    end
  end
end

imsuba = zeros(nPatches, ps-pad*2, ps-pad*2);
imsublowa = zeros(nPatches, ps, ps);
ind = 0;
for sf = test_sf
  for f_iter = 1:numel(f_lst)
    for flip = 0:3
      if ~augment && flip>0, continue; end
      f_info = f_lst(f_iter);
      if f_info.isdir, continue; end
      im = imread(fullfile(dataDir, f_info.name));
      im = rgb2ycbcr(im);
      im = im(:,:,1);
      switch flip
        case 1
          im = im(:,end:-1:1);
        case 2
          im = im(end:-1:1,:);
        case 3
          im = im(end:-1:1, end:-1:1);
        otherwise
      end

      imhigh = modcrop(im, test_sf);
      imhigh = single(imhigh)/255;
      imlow = imresize(imhigh, 1/test_sf, 'bicubic');
      imlow = imresize(imlow, size(imhigh), 'bicubic');
      imlow = max(16.0/255, min(235.0/255, imlow));

      if diff, imhigh=imhigh-imlow; end;
      for rot =0:3
        if ~augment && rot>0, continue; end
        for j = 1:stride:size(imhigh,1)-ps+1
          for k = 1:stride:size(imhigh,2)-ps+1
            imsub = imhigh(j+pad:j+ps-1-pad,k+pad:k+ps-1-pad);
            imsublow = imlow(j:j+ps-1,k:k+ps-1);
            if rot
              imsub=rot90(imsub, rot);
              imsublow = rot90(imsublow,rot);
            end
            ind = ind + 1;
            imsuba(ind,:,:,:)=imsub;
            imsublowa(ind,:,:,:)=imsublow;
          end
        end
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
