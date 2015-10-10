function stats = rcn_train_dag(net, imdb, getBatch, varargin)
%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.problems = {struct('type', 'SR', 'sf', 3)};

opts.dropout = 0;
opts.recursive = 1;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.numEpochs = 300;
opts.batchSize = 64 ;
opts.numSubBatches = 1 ;
opts.learningRate = 0.0005; %3759 - 0.00001, before that 0.0001
opts.weightDecay = 0.0001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp_free') ;
opts.evalDir = fullfile('data','Set5');
opts.prefetch = false ;
opts.momentum = 0.9 ;
opts.derOutputs = {'objective', 1} ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.extractStatsFn = @extractStats ;
opts.pad = 0;
opts.resid = 1;
opts.gradRange = 10000;
opts.useBnorm = false;
opts.testPath = [];
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

state.getBatch = getBatch ;
evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputs)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end
stats = [] ;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
  if exist(opts.memoryMapFile)
    delete(opts.memoryMapFile) ;
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('resuming by loading epoch %d\n', start) ;
  [net, stats] = loadState(modelPath(start)) ;
end

for epoch=start+1:opts.numEpochs

  % train one epoch
  state.epoch = epoch ;
  state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  state.val = opts.val ;
  state.imdb = imdb ;

  if numGpus <= 1
    stats.train(epoch) = process_epoch(net, state, opts, 'train') ;
    stats.val(epoch) = process_epoch(net, state, opts, 'val') ;
  else
    savedNet = net.saveobj() ;
    spmd
      net_ = dagnn.DagNN.loadobj(savedNet) ;
      stats_.train = process_epoch(net_, state, opts, 'train') ;
      stats_.val = process_epoch(net_, state, opts, 'val') ;
      if labindex == 1, savedNet_ = net_.saveobj() ; end
    end
    net = dagnn.DagNN.loadobj(savedNet_{1}) ;
    stats__ = accumulateStats(stats_) ;
    stats.train(epoch) = stats__.train ;
    stats.val(epoch) = stats__.val ;
  end
  
  if numel(opts.gpus)>0, net.move('gpu'); end
  backupmode = net.mode;
  net.mode = 'test';
  [baseline_psnr, stats.test(epoch)] = evalTest(epoch, opts, net);
  net.mode = backupmode;
  net.reset();
  if numel(opts.gpus)>0, net.move('cpu'); end 

  % save
  if ~evaluateMode
    saveState(modelPath(epoch), net, stats) ;
  end

  % test
  if numel(opts.testPath) > 0
      im = imread(opts.testPath);
      im = rgb2ycbcr(im);
      im = im(:,:,1);
      sf = 3;
      imhigh = modcrop(im, sf);
      imhigh = single(imhigh)/255;
      imlow = imresize(imhigh, 1/sf, 'bicubic');
      imlow = imresize(imlow, size(imhigh), 'bicubic');
      imlow = max(16.0/255, min(235.0/255, imlow));
      inputs = {'input', imlow,'label', imhigh};
      net.eval(inputs);
      impred = imlow + net.layers(end).block.lastPred;
  else
      imhigh = []; imlow = []; impred = [];
  end
  
  sfigure(1) ; clf ;
  values = [] ;
  leg = {} ;
  for s = {'train', 'val'}
    s = char(s) ;
    for f = setdiff(fieldnames(stats.train)', {'num', 'time'})
      f = char(f) ;
      leg{end+1} = sprintf('%s (%s)', f, s) ;
      values(end+1,:) = [stats.(s).(f)] ;
    end
  end
  subplot(1,2,1) ; plot(1:epoch, values') ;
  legend(leg{:}) ; xlabel('epoch') ; ylabel('metric') ;
  grid on;
  subplot(1,2,2) ; plot(1:epoch, [repmat(baseline_psnr, 1, epoch); stats.test]') ;
  %legend({'Baseline (Set5)', 'Ours (Set5)'}) ; 
  xlabel('epoch') ; ylabel('PSNR') ; title(sprintf('Best PSNR (dropout: %d, recursive: %d) : %f',opts.dropout, opts.recursive,  max(stats.test)));
  grid on ;
%   subplot(2,3,4) ; imshow(imhigh);
%   subplot(2,3,5) ; imshow(imlow);
%   subplot(2,3,6) ; imshow(impred);
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
end

% -------------------------------------------------------------------------
function stats = process_epoch(net, state, opts, mode)
% -------------------------------------------------------------------------

if strcmp(mode,'train')
  state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  if strcmp(mode,'train')
    state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
  end
end
if numGpus > 1
  mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
  mmap = [] ;
end

stats.time = 0 ;
stats.scores = [] ;
subset = state.(mode) ;
start = tic ;
num = 0 ;

for t=1:opts.batchSize:numel(subset)
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;

  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    inputs = state.getBatch(opts, state.imdb, batch) ;

    if opts.prefetch
      if s == opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      state.getBatch(state.imdb, nextBatch) ;
    end

    if strcmp(mode, 'train')
      net.accumulateParamDers = 1;%(s ~= 1) ;
      for i=1:numel(net.params)
          net.params(i).der = [];
      end  
      net.eval(inputs, opts.derOutputs) ;
    else
      net.eval(inputs) ;
    end
  end

  % extract learning stats
  stats = opts.extractStatsFn(net) ;

  % accumulate gradient
  if strcmp(mode, 'train')
    if ~isempty(mmap)
      write_gradients(mmap, net) ;
      labBarrier() ;
    end
    state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
  end

  % print learning statistics
  time = toc(start) ;
  stats.num = num ;
  stats.time = toc(start) ;

  fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
    mode, ...
    state.epoch, ...
    fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize), ...
    stats.num/stats.time * max(numGpus, 1)) ;
  for f = setdiff(fieldnames(stats)', {'num', 'time'})
    f = char(f) ;
    fprintf(' %s:%.3f', f, stats.(f)) ;
  end
  fprintf('\n') ;
end

net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulate_gradients(state, net, opts, batchSize, mmap)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  if ~isempty(mmap)
    tmp = zeros(size(mmap.Data(labindex).(net.params(i).name)), 'single') ;
    for g = setdiff(1:numel(mmap.Data), labindex)
      tmp = tmp + mmap.Data(g).(net.params(i).name) ;
    end
    net.params(i).der = net.params(i).der + tmp ;
  end
  lr = state.learningRate;
  mult = lr * net.params(i).learningRate / batchSize;
  net.params(i).der = min(max(net.params(i).der, -opts.gradRange/mult), opts.gradRange/mult);
  thisDecay = opts.weightDecay * net.params(i).weightDecay ;
  
  momentum_prev = state.momentum{i};
  state.momentum{i} = opts.momentum * state.momentum{i} ...
    - lr * net.params(i).learningRate * ...
      thisDecay * net.params(i).value ...
    - lr * net.params(i).learningRate * (1 / batchSize) * net.params(i).der ;
  
  %Nesterov
  net.params(i).value = net.params(i).value ...
                        - opts.momentum * momentum_prev ...
                        + (1 + opts.momentum) * state.momentum{i};
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.params)
  format(end+1,1:3) = {'single', size(net.params(i).value), net.params(i).name} ;
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

stats = struct() ;

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;

      if g == 1
        stats.(s).(f) = 0 ;
      end
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) (isa(x,'dagnn.Loss') || isa(x,'dagnn.EuclidLoss')), {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).name) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net, stats)
% -------------------------------------------------------------------------
net_ = net ;
net = net_.saveobj() ;
save(fileName, 'net', 'stats') ;

% -------------------------------------------------------------------------
function [net, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;

% -------------------------------------------------------------------------
function [eval_base, eval_ours] = evalTest(epoch, opts, net)
% -------------------------------------------------------------------------    
% Evaluation
%fid = fopen(opts.fname,'w');
%fprintf(fid, 'Epoch: %d\n', epoch);
f_lst = dir(opts.evalDir);
eval_base = zeros(numel(opts.problems),1);
eval_ours = zeros(numel(opts.problems),1);

f_n = 0;
printPic = true;
for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
    if f_info.isdir, continue; end
    f_n = f_n + 1;
    im = imread(fullfile(opts.evalDir, f_info.name));
    im = rgb2ycbcr(im);
    im = im(:,:,1);

    if printPic && f_n==1, imwrite(im, 'GT.bmp'); end
    
    for problem_iter = 1:numel(opts.problems)
        problem = opts.problems{problem_iter};
        
        % preprocess
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
        if numel(opts.gpus) > 0
            imlow = gpuArray(imlow);
            imhigh = gpuArray(imhigh);
        end
        
        % predict
        inputs = {'input', imlow, 'label', imhigh };
        net.eval(inputs);
        impred = net.layers(end).block.lastPred;
        
        % post process
        switch problem.type
            case 'SR'
                impred = shave(impred, [problem.sf, problem.sf]);
                imhigh = shave(imhigh, [problem.sf, problem.sf]);
                imlow = shave(imlow, [problem.sf, problem.sf]);
            case 'JPEG'
                %
            case 'DENOISE'
                %
        end
        imhigh = imhigh(opts.pad+1:end-opts.pad,opts.pad+1:end-opts.pad);
        imlow = imlow(opts.pad+1:end-opts.pad,opts.pad+1:end-opts.pad);
        if opts.resid, impred = impred+imlow; end
        impred = uint8(impred * 255);
        imlow = uint8(imlow * 255);
        imhigh = uint8(imhigh * 255);
        
        % evaluate
        evalType = 'PSNR';
        if isfield(problem, 'evalType')
            evalType = problem.evalType;
        end
        switch evalType
            case 'PSNR'
                eval_base(problem_iter) = eval_base(problem_iter) + gather(compute_psnr(imhigh,imlow));
                eval_ours(problem_iter) = eval_ours(problem_iter) + gather(compute_psnr(imhigh,impred));
        end
        
        if printPic && f_n == 1
            imwrite(gather(imlow),  strcat(problem.type,'_low.bmp'));
            imwrite(gather(impred), strcat(problem.type,'_pred.bmp'));
        end
    end
end
for problem_iter = 1:numel(opts.problems) 
    problem = opts.problems{problem_iter};
    eval_base(problem_iter) = eval_base(problem_iter) / f_n;
    eval_ours(problem_iter) = eval_ours(problem_iter) / f_n;
%    fprintf(fid,'%f\t%f\t%f\t%s\n', eval_ours(problem_iter)-eval_base(problem_iter), eval_base(problem_iter), eval_ours(problem_iter), problem.type);
end
%fclose(fid);

function h = sfigure(h)
% SFIGURE  Create figure window (minus annoying focus-theft).
%
% Usage is identical to figure.
%
% Daniel Eaton, 2005
%
% See also figure

if nargin>=1
    if ishandle(h)
        set(0, 'CurrentFigure', h);
    else
        h = figure(h);
    end
else
    h = figure;
end      

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;                                                                                                                                               