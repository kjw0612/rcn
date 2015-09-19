function [net, info] = rcn_train(net, imdb, getBatch, varargin)
% CNN_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastic gradient
%    descent with momentum to train a CNN for image classification.
%    It can be used with different datasets by providing a suitable
%    getBatch function.

opts.problems = {struct('type', 'SR', 'sf', 3)};

opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 10000;
opts.batchSize = 64 ;
opts.useGpu = true ;
opts.learningRate = 0.0005; %3759 - 0.00001, before that 0.0001
opts.continue = true ;
opts.expDir = fullfile('data','exp_free') ;
opts.evalDir = fullfile('data','Set5');
opts.conserveMemory = false ;
opts.sync = false ;
opts.prefetch = false ;
opts.weightDecay = 0.0001 ;
opts.momentum = 0.9 ;
opts.errorType = 'euclidean';
opts.plotDiagnostics = false ;
opts.pad = 10;
opts.resid = 1;
opts.fname = 'result.txt'; % result to write such as PSNRs
opts.gradRange = 10000;
opts.useBnorm = false;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

for i=1:numel(net.layers)
    if strcmp(net.layers{i}.type,'conv')
        net.layers{i}.filtersMomentum = zeros(size(net.layers{i}.filters), ...
            class(net.layers{i}.filters)) ;
        net.layers{i}.biasesMomentum = zeros(size(net.layers{i}.biases), ...
            class(net.layers{i}.biases)) ; %#ok<*ZEROLIKE>
        if ~isfield(net.layers{i}, 'filtersLearningRate')
            net.layers{i}.filtersLearningRate = 1 ;
        end
        if ~isfield(net.layers{i}, 'biasesLearningRate')
            net.layers{i}.biasesLearningRate = 1;
        end
        if ~isfield(net.layers{i}, 'filtersWeightDecay')
            net.layers{i}.filtersWeightDecay = 1 ;
        end
        if ~isfield(net.layers{i}, 'biasesWeightDecay')
            net.layers{i}.biasesWeightDecay = 1 ;
        end
    end
    if strcmp(net.layers{i}.type,'prelu')
        net.layers{i}.slopesMomentum = zeros(size(net.layers{i}.slopes), ...
            class(net.layers{i}.slopes)) ;
        if ~isfield(net.layers{i}, 'slopesLearningRate')
            net.layers{i}.slopesLearningRate = 1 ;
        end
    end
end

if opts.useGpu
    net = vl_simplenn_move(net, 'gpu') ;
    for i=1:numel(net.layers)
        if strcmp(net.layers{i}.type,'conv')
            net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum) ;
            net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum) ;
        end
        if strcmp(net.layers{i}.type,'prelu')
            net.layers{i}.slopesMomentum = gpuArray(net.layers{i}.slopesMomentum) ;
        end
    end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

rng(0) ;

if opts.useGpu
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end

info.train.objective = [] ;
info.train.error = [] ;
info.train.topFiveError = [] ;
info.train.speed = [] ;
info.val.objective = [] ;
info.val.error = [] ;
info.val.topFiveError = [] ;
info.val.speed = [] ;
info.test.error = cell(numel(opts.problems),1);
for problem_iter = 1:numel(opts.problems)
    info.test.error{problem_iter}.base = [];
    info.test.error{problem_iter}.ours = [];
end

lr = 0 ;
res = [] ;
for epoch=1:opts.numEpochs
    prevLr = lr ;
    lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    gradRange = opts.gradRange(min(epoch, numel(opts.gradRange))) ;
    
    % fast-forward to where we stopped
    modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
    modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
    if opts.continue
        if exist(modelPath(epoch),'file')
            if epoch == opts.numEpochs
                load(modelPath(epoch), 'net', 'info') ;
            end
            continue ;
        end
        if epoch > 1
            fprintf('resuming by loading epoch %d\n', epoch-1) ;
            load(modelPath(epoch-1), 'net', 'info') ;
        end
    end
    
    train = opts.train(randperm(numel(opts.train))) ;
    val = opts.val ;
    
    info.train.objective(end+1) = 0 ;
    info.train.error(end+1) = 0 ;
    info.train.topFiveError(end+1) = 0 ;
    info.train.speed(end+1) = 0 ;
    info.val.objective(end+1) = 0 ;
    info.val.error(end+1) = 0 ;
    info.val.topFiveError(end+1) = 0 ;
    info.val.speed(end+1) = 0 ;
    
    % reset momentum if needed
    if prevLr ~= lr
        fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr) ;
        for l=1:numel(net.layers)
            if strcmp(net.layers{l}.type, 'conv')
                net.layers{l}.filtersMomentum = 0 * net.layers{l}.filtersMomentum ;
                net.layers{l}.biasesMomentum = 0 * net.layers{l}.biasesMomentum ;
            end
            if strcmp(net.layers{l}.type, 'prelu')
                net.layers{l}.slopesMomentum = 0 * net.layers{l}.slopesMomentum ;
            end
        end
    end
    
    for t=1:opts.batchSize:numel(train)
        % get next image batch and labels
        batch = train(t:min(t+opts.batchSize-1, numel(train))) ;
        batch_time = tic ;
        fprintf('training: epoch %02d: batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize)) ;
        [im, labels] = getBatch(imdb, batch) ;
        if opts.prefetch
            nextBatch = train(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(train))) ;
            getBatch(imdb, nextBatch) ;
        end
        if opts.useGpu
            im = gpuArray(im) ;
        end
        
        % backprop
        net.layers{end}.class = labels ;
        res = vl_simplenn(net, im, one, res, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync) ;
        
        % gradient step
        maxGrad = 0;
        for l=1:numel(net.layers)
            if strcmp(net.layers{l}.type, 'conv')
                mult1 = (lr * net.layers{l}.filtersLearningRate) / numel(batch);
                mult2 = (lr * net.layers{l}.biasesLearningRate) / numel(batch);
                maxGrad = max(max(res(l).dzdw{1}(:)*mult1) * lr, maxGrad);
                maxGrad = max(max(res(l).dzdw{2}(:)*mult2), maxGrad);
                res(l).dzdw{1} = min(max(res(l).dzdw{1}, -gradRange/mult1),gradRange/mult1);
                res(l).dzdw{2} = min(max(res(l).dzdw{2}, -gradRange/mult2),gradRange/mult2);
                
                net.layers{l}.filtersMomentum = ...
                    opts.momentum * net.layers{l}.filtersMomentum ...
                    - (lr * net.layers{l}.filtersLearningRate) * ...
                    (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
                    - (lr * net.layers{l}.filtersLearningRate) / numel(batch) * res(l).dzdw{1} ;
                
                net.layers{l}.biasesMomentum = ...
                    opts.momentum * net.layers{l}.biasesMomentum ...
                    - (lr * net.layers{l}.biasesLearningRate) * ....
                    (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
                    - (lr * net.layers{l}.biasesLearningRate) / numel(batch) * res(l).dzdw{2} ;
                
                net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
                net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
            end
            if strcmp(net.layers{l}.type, 'prelu')
                
                net.layers{l}.slopesMomentum = ...
                    opts.momentum * net.layers{l}.slopesMomentum ...
                    - (lr * net.layers{l}.slopesLearningRate) * res(l).dzdw;
                
                net.layers{l}.slopes = net.layers{l}.slopes + net.layers{l}.slopesMomentum ;
                %          net.layers{l}.slopes
            end
        end
        
        % print information
        batch_time = toc(batch_time) ;
        speed = numel(batch)/batch_time ;
        info.train = updateError(opts, info.train, net, res, batch_time) ;
        
        fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
        n = t + numel(batch) - 1 ;
        fprintf(' lr %.4f  err %.4f maxGrad %.4f gradRange %.4f', ...
            lr, info.train.error(end)/n*100, maxGrad, gradRange) ;
        fprintf('\n') ;
        
        % debug info
        if opts.plotDiagnostics
            sfigure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ; waitforbuttonpress;
        end
    end % next batch
    
    % evaluation on validation set
    for t=1:opts.batchSize:numel(val)
        batch_time = tic ;
        batch = val(t:min(t+opts.batchSize-1, numel(val))) ;
        fprintf('validation: epoch %02d: processing batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;
        [im, labels] = getBatch(imdb, batch) ;
        if opts.prefetch
            nextBatch = val(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(val))) ;
            getBatch(imdb, nextBatch) ;
        end
        if opts.useGpu
            im = gpuArray(im) ;
        end
        
        net.layers{end}.class = labels ;
        res = vl_simplenn(net, im, [], res, ...
            'disableDropout', true, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync) ;
        
        % print information
        batch_time = toc(batch_time) ;
        speed = numel(batch)/batch_time ;
        info.val = updateError(opts, info.val, net, res, batch_time) ;
        
        fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
        n = t + numel(batch) - 1 ;
        fprintf(' err %.1f err5 %.1f', ...
            info.val.error(end)/n*100, info.val.topFiveError(end)/n*100) ;
        fprintf('\n') ;
    end
    
    [eval_base, eval_ours] = evalTest(epoch, opts, net);
    
    % save max
    max_path = fullfile(opts.expDir, 'best.mat');
    fid = fopen(max_path);
    max_eval = 0;
    if fid ~= -1
        load(max_path, 'max_eval');
        fclose(fid);
    end
    if max_eval < mean(gather(eval_ours))
        max_eval = mean(gather(eval_ours));
        save(max_path, 'net', 'info', 'epoch', 'opts', 'max_eval');
    end
    
    % save
    info.train.objective(end) = info.train.objective(end) / numel(train) ;
    info.train.error(end) = info.train.error(end) / numel(train) ;
    info.train.topFiveError(end) = info.train.topFiveError(end) / numel(train) ;
    info.train.speed(end) = numel(train) / info.train.speed(end) ;
    info.val.objective(end) = info.val.objective(end) / numel(val) ;
    info.val.error(end) = info.val.error(end) / numel(val) ;
    info.val.topFiveError(end) = info.val.topFiveError(end) / numel(val) ;
    info.val.speed(end) = numel(val) / info.val.speed(end) ;
    save(modelPath(epoch), 'net', 'info') ;
    
    sfigure(1) ; clf ;
    subplot(2,numel(opts.problems),1) ;
    %skip several epochs for better visualization
    plot(1:epoch, info.train.objective(1:end), 'k') ; hold on ;
    plot(1:epoch, info.val.objective(1:end), 'b') ;
    xlabel('training epoch') ; ylabel('energy') ;
    grid on ;
    h=legend('train', 'val') ;
    set(h,'color','none');
    title('objective') ;
    
    for problem_iter = 1:numel(opts.problems)
        subplot(2,numel(opts.problems),problem_iter+numel(opts.problems)) ;
        info.test.error{problem_iter}.base(end+1) = gather(eval_base(problem_iter));
        info.test.error{problem_iter}.ours(end+1) = gather(eval_ours(problem_iter));
        plot(1:epoch, info.test.error{problem_iter}.base, 'k') ; hold on ;
        plot(1:epoch, info.test.error{problem_iter}.ours, 'b') ;
        h=legend('Baseline', 'Ours') ;
        grid on ;
        xlabel('training epoch') ; ylabel('error') ;
        set(h,'color','none') ;
        title(['Evaluation of ',opts.problems{problem_iter}.type]);
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
end

% -------------------------------------------------------------------------
function info = updateError(opts, info, net, res, speed)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
sz = size(predictions) ;
n = prod(sz(1:2)) ;

labels = net.layers{end}.class ;
info.objective(end) = info.objective(end) + sum(sum(sum(double(gather(res(end).x))))) ;
info.speed(end) = info.speed(end) + speed ;
switch opts.errorType
    case 'multiclass'
        [~,predictions] = sort(predictions, 3, 'descend') ;
        error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
        info.error(end) = info.error(end) +....
            sum(sum(sum(error(:,:,1,:))))/n ;
        info.topFiveError(end) = info.topFiveError(end) + ...
            sum(sum(sum(min(error(:,:,1:5,:),[],3))))/n ;
    case 'binary'
        error = bsxfun(@times, predictions, labels) < 0 ;
        info.error(end) = info.error(end) + sum(error(:))/n ;
    case 'euclidean'
        error = (predictions - labels).^2;
        info.error(end) = info.error(end) + sum(error(:))/n ;
end

% -------------------------------------------------------------------------
function h = sfigure(h)
% -------------------------------------------------------------------------
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
end

% -------------------------------------------------------------------------
function [eval_base, eval_ours] = evalTest(epoch, opts, net)
% -------------------------------------------------------------------------    
% Evaluation
fid = fopen(opts.fname,'w');
fprintf(fid, 'Epoch: %d\n', epoch);
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
        if opts.useGpu
            imlow = gpuArray(imlow);
            imhigh = gpuArray(imhigh);
        end
        
        % predict
        net2.layers = net.layers(1:end-1);
        res2 = vl_simplenn(net2, imlow, [], [], 'disableDropout', true, 'conserveMemory', true) ;
        impred = res2(end).x;
        
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
    fprintf(fid,'%f\t%f\t%f\t%s\n', eval_ours(problem_iter)-eval_base(problem_iter), eval_base(problem_iter), eval_ours(problem_iter), problem.type);
end
fclose(fid);
