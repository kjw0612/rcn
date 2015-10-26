function [net, derOutputs] = rcn_init_dag(opts)
% define net
net = dagnn.DagNN();

convBlock = dagnn.Conv('size', [3,3,1,opts.filterSize], 'hasBias', true, 'init', [1, 0], 'pad', 1);
net.addLayer('conv1', convBlock, {'input'}, {'x1'}, {'filters1', 'biases1'});
net.addLayer('relu1', dagnn.ReLU(), {'x1'}, {'x2'}, {}) ;
x = 2;
if opts.useBnorm
    net.addLayer('bnorm1', dagnn.BatchNorm('ndim', opts.filterSize, 'nbatch', opts.train.batchSize), {'x2'}, {'x3'}, {'gamma1','beta1'}) ;
	x = x + 1;
end

preds = {};


for i = 2 : opts.depth - 2
    convBlock = dagnn.Conv('size', [3,3,opts.filterSize,opts.filterSize], 'hasBias', true, 'init', [1, 0], 'pad', 1, 'initIdentity', 1);
    if opts.recursive && i <= opts.depth - 2 && i >= 3
        net.addLayer(['conv',num2str(i)], convBlock, {['x',num2str(x)]}, {['x',num2str(x+1)]}, {'filters_share', 'biases_share'});
    else
        net.addLayer(['conv',num2str(i)], convBlock, {['x',num2str(x)]}, {['x',num2str(x+1)]}, {['filters',num2str(i)], ['biases',num2str(i)]});
    end
    if opts.dropout
     x = x + 1;
     net.addLayer(['dropout',num2str(i)], dagnn.DropOut('rate', 0.2), {['x',num2str(x)]}, {['x',num2str(x+1)]}, {}) ;
    end
    x = x + 1;
    net.addLayer(['relu',num2str(i)], dagnn.ReLU(), {['x',num2str(x)]}, {['x',num2str(x+1)]}, {}) ;
    x = x + 1;
    if opts.useBnorm
        net.addLayer(['bnorm',num2str(i)], dagnn.BatchNorm('ndim', opts.filterSize, 'nbatch', opts.train.batchSize), {['x',num2str(x)]}, {['x',num2str(x+1)]}, {['gamma',num2str(i)], ['beta',num2str(i)]}) ;
        x = x + 1;
    end
    if opts.deep_supervise
      if i < opts.depth - 1
          init = [0.001, 0.5];
          net.addLayer(['conv_outh',num2str(i)], convBlock, {['x',num2str(x)]}, {['x',num2str(x+1)]}, {['filters',num2str(opts.depth-1)], ['biases',num2str(opts.depth-1)]});
          x=x+1;
          if opts.resid, init(2)=0; end
          convBlock = dagnn.Conv('size', [3,3,opts.filterSize,1], 'hasBias', true, 'init', init, 'pad', 1);        
          net.addLayer(sprintf('conv_out%d',i), convBlock, {sprintf('x%d',x)}, {sprintf('prediction%d',i)}, {['filters',num2str(opts.depth)], ['biases',num2str(opts.depth)]});
          net.addLayer(sprintf('objective%d',i), dagnn.EuclidLoss(), ...
               {sprintf('prediction%d',i),'label'}, sprintf('objective%d',i)) ;
          preds{end+1} = sprintf('prediction%d',i);
      end
    end
end
%net.params(net.getParamIndex('filters_share')).learningRate =  1/sqrt(opts.depth - 4);
%net.params(net.getParamIndex('biases_share')).learningRate = 1/sqrt(opts.depth - 4);

% init = [0.001, 0.5];
% if opts.resid, init(2)=0; end
% convBlock = dagnn.Conv('size', [3,3,opts.filterSize,1], 'hasBias', true, 'init', init, 'pad', 1);
% net.addLayer(['conv',num2str(opts.depth)], convBlock, {['x',num2str(x)]}, {'prediction'}, {['filters',num2str(opts.depth)], ['biases',num2str(opts.depth)]});

convBlock = dagnn.Conv('size', [1,1,opts.depth-3,1], 'hasBias', false);
net.addLayer(sprintf('concat'), dagnn.Concat(), preds, {'concat_pred'}, {}); 
net.addLayer(sprintf('ensemble'), convBlock, {'concat_pred'}, {'prediction'}, {'ensemble_weight'});
 net.addLayer('objective', dagnn.EuclidLoss(), ...
              {'prediction','label'}, 'objective') ;

derOutputs =  {'objective', 1};
if opts.deep_supervise
  for i=2:opts.depth-3
      derOutputs{end+1}=sprintf('objective%d',i);
      derOutputs{end+1}=1;
  end
end