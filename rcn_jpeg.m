function [net, info] = rcn_jpeg(varargin)
% JPEG Artifact Reduction
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
opts.qs = 20; % JPEG quality for training set
opts.resid = 1;
opts.depth = 10; % 10 optimal
opts.continue = 0;
opts.gradRange = 1e-4;
opts.sync = true;
opts.dataDir = 'data/91'; % 'data/291' for bsds 

opts.filterSize = 64; % number of filters
opts = vl_argparse(opts, varargin);

exp_name = sprintf('exp_q%s_resid%d_depth%d', mat2str(opts.qs), opts.resid, opts.depth);
opts.expDir = fullfile('data/exp',exp_name);
rep=20;
opts.learningRate = [0.1*ones(1,rep) 0.01*ones(1,rep) 0.001*ones(1,rep) 0.0001*ones(1,rep)];%*0.99 .^ (0:500);
opts.gradRange = [1e-4];
%opts.learningRate = 0.0000001;
%opts.learningRate = [0.01*ones(1,10) 0.001]; % 0.0005 originally
if ~exist('data/result', 'dir'), mkdir('data/result'); end
opts.fname = sprintf('data/result/%s.txt',exp_name);
if opts.depth <= 20
    opts.batchSize = 64;
else
    opts.batchSize = 32;
end;

%if numel(opts.sfs) == 1
%    opts.weightDecay = 0.0001;
%elseif numel(opts.sfs) == 2
%    opts.weightDecay = 0.0001;
%elseif numel(opts.sfs) == 3
%    opts.weightDecay = 0.0001;
%else
%    opts.weightDecay = 0.0001;
%end;
opts.weightDecay = 0.0001;

opts.numEpochs = numel(opts.learningRate);
% if ~opts.diff
%     opts.numEpochs = 1000;
% %    opts.learningRate = 0.0001;
%     opts.learningRate = [0.1*ones(1,40) 0.01*ones(1,40) 0.001*ones(1,40) 0.0001*ones(1,40) 0.00001*ones(1,40)];%*0.99 .^ (0:500);
% end
%opts.pad = opts.depth;
opts.pad = 0;
opts.plotDiagnostics = 0 ;

if ~exist('data/trainData', 'dir'), mkdir('data/trainData'); end
if ~opts.continue || ~exist(sprintf('data/trainData/%s.mat',exp_name),'file')
    imdb = generateData(opts.dataDir, opts.qs, opts.depth, opts.pad, opts.resid);
    save(sprintf('data/trainData/%s.mat',exp_name), 'imdb');
else
    load(sprintf('data/trainData/%s.mat',exp_name));
end

% subplot(121)
% vl_imarraysc(imdb.data(:,:,1:100))
% subplot(122)
% vl_imarraysc(imdb.labels(:,:,1:100))

% define net
net.layers = {} ;
%                           'filters', sqrt(2/9)*randn(3,3,1,opts.filterSize, 'single'), ...
if opts.depth > 1
    net.layers{end+1} = struct('type', 'conv', ...
                               'filters',sqrt(2/9)*randn(3,3,1,opts.filterSize, 'single'), ...
                               'biases', zeros(1, opts.filterSize, 'single'), ...
                               'stride', 1, ...
                               'pad', 1);
    net.layers{end+1} = struct('type', 'relu');
end
for i=1:opts.depth - 2
%                               'filters', 0.0589*randn(3,3,opts.filterSize,opts.filterSize, 'single'), ...
    net.layers{end+1} = struct('type', 'conv', ...
                               'filters', sqrt(2/9/opts.filterSize)*randn(3,3,opts.filterSize,opts.filterSize, 'single'), ...
                               'biases', zeros(1, opts.filterSize, 'single'), ...
                               'stride', 1, ...
                               'pad', 1) ;
    net.layers{end+1} = struct('type', 'relu');
end
if opts.resid
    bias_diff = 0; %if diff, it's centered aroun zero
else
    bias_diff = 0.5; % if not diff, it's centered around 0.5 which is the average DC.
end;
%                           'filters', sqrt(2/9/opts.filterSize)*randn(3,3,opts.filterSize,1, 'single'),...
if opts.depth > 1
    net.layers{end+1} = struct('type', 'conv', ...
                               'filters', 0.001*sqrt(2/9/opts.filterSize)*randn(3,3,opts.filterSize,1, 'single'),...
                               'biases', bias_diff + zeros(1,1,'single'), ...
                               'stride', 1, ...
                               'pad', 1);
else
    net.layers{end+1} = struct('type', 'conv', ...
                               'filters', 0.001*sqrt(2/9/1)*randn(3,3,1,1, 'single'),...
                               'biases', bias_diff + zeros(1,1,'single'), ...
                               'stride', 1, ...
                               'pad', 1);
end
net.layers{end+1} = struct('type', 'euclidloss') ;

opts = rmfield(opts, {'depth', 'filterSize', 'dataDir'}); % remove irrelavant fields
[net, info] = rcn_train_jpeg(net, imdb, @getBatch, opts) ;

function imdb = generateData(dataDir, qs, depth, pad, diff)
f_lst = dir(fullfile(dataDir, '*.*'));
ps = (2*depth+1); % patch size
stride = ps;%ps - 2*pad;

nPatches = 0;
for q=qs
    for i = 3:size(f_lst,1)
        for flip = 0
            im = imread(fullfile(dataDir, f_lst(i).name));
            im = rgb2ycbcr(im);
            im = im(:,:,1);
            switch flip
                case 1
                    im = im(:,end:-1:1);
                case 2
                    im = im(end:-1:1,:);
                case 3
                    im = im(end:-1:1,end:-1:1);
                otherwise
            end
            imhigh = im;
            for j = 1:stride:size(imhigh,1)-ps+1
                for k = 1:stride:size(imhigh,2)-ps+1
                    nPatches = nPatches +1;
                end
            end
        end
    end
end

imsuba = zeros(nPatches, ps-pad*2, ps-pad*2);
imsublowa = zeros(nPatches, ps, ps);
ind = 0;
for q=qs
    for i = 3:size(f_lst,1)
        for flip = 0
            im = imread(fullfile(dataDir, f_lst(i).name));
            im = rgb2ycbcr(im);
            im = im(:,:,1);
            switch flip
                case 1
                    im = im(:,end:-1:1);
                case 2
                    im = im(end:-1:1,:);
                case 3
                    im = im(end:-1:1,end:-1:1);
                otherwise
            end
            imhigh = im;
            imhigh = single(imhigh)/255;
            
            imwrite(imhigh, 'data/_temp.jpg', 'Quality', q);
            imlow = imread('data/_temp.jpg');
            imlow = single(imlow)/255;
            delete('data/_temp.jpg');
            
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
%save('data.mat', 'imsuba', 'imsublowa');
% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(:,:,:,batch) ;