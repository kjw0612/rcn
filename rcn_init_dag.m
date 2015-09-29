function net = rcn_init_dag(opts)
% define net
net = dagnn.DagNN() ;

convBlock = dagnn.Conv('size', [3,3,1,opts.filterSize], 'hasBias', true, 'init', [1, 0], 'pad', 1);
net.addLayer('conv1', convBlock, {'input'}, {'x1'}, {'filters1', 'biases1'});
net.addLayer('relu1', dagnn.ReLU(), {'x1'}, {'x2'}, {}) ;
x = 2;
if opts.useBnorm
    net.addLayer('bnorm1', dagnn.BatchNorm(), {'x2'}, {'x3'}, {}) ;
	x = x + 1;
end
for i = 2 : opts.depth - 1
    convBlock = dagnn.Conv('size', [3,3,opts.filterSize,opts.filterSize], 'hasBias', true, 'init', [1, 0], 'pad', 1);
    net.addLayer(['conv',num2str(i)], convBlock, {['x',num2str(x)]}, {['x',num2str(x+1)]}, {['filters',num2str(i)], ['biases',num2str(i)]});
    x = x + 1;
    net.addLayer(['relu',num2str(i)], dagnn.ReLU(), {['x',num2str(x)]}, {['x',num2str(x+1)]}, {}) ;
    x = x + 1;
    if opts.useBnorm
        net.addLayer(['bnorm',num2str(i)], dagnn.BatchNorm(), {['x',num2str(x)]}, {['x',num2str(x+1)]}, {}) ;
        x = x + 1;
    end
    
if i >= 5    
convBlock = dagnn.Conv('size', [3,3,opts.filterSize,1], 'hasBias', true, 'init', [0.001, 0], 'pad', 1);
net.addLayer(sprintf('pred%d',i), convBlock, {sprintf('x%d',x)}, {sprintf('pred%d',i)}, {sprintf('filters_pred%d', i), sprintf('biases_pred%d', i)});
net.addLayer(sprintf('objective%d',i), dagnn.EuclidLoss(), ...
             {sprintf('pred%d',i), 'label'},  sprintf('objective%d',i)) ;
end
end
init = [0.001, 0.5];
if opts.resid, init(2)=0; end
net.addLayer('concat', dagnn.Concat(), {'pred5', 'pred6', 'pred7', 'pred8', 'pred9'}, {'pred_temp'});
convBlock = dagnn.Conv('size', [5,5,5,512], 'hasBias', true, 'init', [1/5 0], 'pad', 2);
net.addLayer('conv91', convBlock, {'pred_temp'}, {'pred_temp2'}, {'filters91', 'biases91'}) ;
net.addLayer('relu91', dagnn.ReLU(), {'pred_temp2'}, {'pred_temp3'}, {}) ;
net.addLayer('conv92', dagnn.Conv('size', [1 1 512 512],'hasBias', true,'init',[1, 0]), {'pred_temp3'}, {'pred_temp4'}, {'filters92', 'biases92'}) ;
net.addLayer('relu92', dagnn.ReLU(), {'pred_temp4'}, {'pred_temp5'}, {}) ;
net.addLayer('conv93', dagnn.Conv('size', [1 1 512 1],'hasBias', true,'init',[0.001, 0]), {'pred_temp5'}, {'prediction'}, {'filters93', 'biases93'}) ;
%net.addLayer(['conv',num2str(opts.depth)], convBlock, {['x',num2str(x)]}, {'prediction'}, {['filters',num2str(opts.depth)], ['biases',num2str(opts.depth)]});
%net.addLayer(['conv',num2str(opts.depth)], convBlock, {'pred_temp'}, {'prediction'}, {['filters',num2str(opts.depth)], ['biases',num2str(opts.depth)]});

net.addLayer('objective', dagnn.EuclidLoss(), ...
             {'prediction','label'}, 'objective') ;
