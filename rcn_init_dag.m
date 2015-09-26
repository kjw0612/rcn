function net = rcn_init_dag(opts)
% define net
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{sqrt(2/9)*randn(3,3,1,opts.filterSize, 'single'),...
                                        zeros(1, opts.filterSize, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 1);
net.layers{end+1} = struct('type', 'relu');
for i=1:opts.depth - 2
    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{sqrt(2/9/opts.filterSize)*randn(3,3,opts.filterSize,opts.filterSize, 'single'),...
                                            zeros(1, opts.filterSize, 'single')}}, ...
                               'stride', 1, ...
                               'pad', 1) ;
    net.layers{end+1} = struct('type', 'relu');
end
if opts.resid
    bias_diff = 0; %if diff, it's centered aroun zero
else
    bias_diff = 0.5; % if not diff, it's centered around 0.5 which is the average DC.
end
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.001*sqrt(2/9/opts.filterSize)*randn(3,3,opts.filterSize,1, 'single'),...
                                        bias_diff + zeros(1,1,'single')}},...
                           'stride', 1, ...
                           'pad', 1);
net.layers{end+1} = struct('type', 'euclidloss') ;

% optionally switch to batch normalization
if opts.useBnorm
    d = 1;
    while d+1 < numel(net.layers)
        if strcmp(net.layers{d}.type,'conv')
            net = insertBnorm(net, d);
        end
        d = d + 1;
    end
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.filters, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;