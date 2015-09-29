function net = rcn_init_dag(opts)
% define net
net = dagnn.DagNN() ;

%convBlock = dagnn.Conv('size', [3,3,2,opts.filterSize], 'hasBias', true, 'init', [1, 0], 'pad', 1);
%net.addLayer('conv1', convBlock, {'input'}, {'x1'}, {'filters1', 'biases1'});
%net.addLayer('relu1', dagnn.ReLU(), {'x1'}, {'x2'}, {}) ;
x = 2;
net.addLayer('conv00', dagnn.Conv('size', [1,1,2,512], 'hasBias', true, 'init', [1, 0]), {'input'}, {'x00'}, {'filters00', 'biases00'});
net.addLayer('relu00', dagnn.ReLU(), {'x00'}, {'x02'}, {}) ;
net.addLayer('conv01', dagnn.Conv('size', [1,1,512,512], 'hasBias', true, 'init', [1, 0]), {'x02'}, {'x03'}, {'filters01', 'biases01'});
net.addLayer(['dropout01'], dagnn.DropOut('rate', 0.5), {'x03'}, {'x04'}) ;
net.addLayer('relu01', dagnn.ReLU(), {'x04'}, {'x05'}, {}) ;
net.addLayer('conv02', dagnn.Conv('size', [1,1,512,64], 'hasBias', true, 'init', [1, 0]), {'x05'}, {'x06'}, {'filters02', 'biases02'});
net.addLayer('relu02', dagnn.ReLU(), {'x06'}, {'x2'}, {}) ;
if opts.useBnorm
    net.addLayer('bnorm1', dagnn.BatchNorm(), {'x2'}, {'x3'}, {}) ;
	x = x + 1;
end
for i = 2 : opts.depth %- 1
    convBlock = dagnn.Conv('size', [3,3,opts.filterSize,opts.filterSize], 'hasBias', true, 'init', [1, 0], 'pad', 1);
    net.addLayer(['conv',num2str(i)], convBlock, {['x',num2str(x)]}, {['x',num2str(x+1)]}, {['filters',num2str(i)], ['biases',num2str(i)]});
     x = x + 1;
%      net.addLayer(['dropout',num2str(i)], dagnn.DropOut('rate', 0.5), {['x',num2str(x)]}, {['x',num2str(x+1)]}, {}) ;
%     x = x + 1;
    net.addLayer(['relu',num2str(i)], dagnn.ReLU(), {['x',num2str(x)]}, {['x',num2str(x+1)]}, {}) ;
    x = x + 1;
    if opts.useBnorm
        net.addLayer(['bnorm',num2str(i)], dagnn.BatchNorm(), {['x',num2str(x)]}, {['x',num2str(x+1)]}, {}) ;
        x = x + 1;
    end
end
init = [0.001, 0.5];
if opts.resid, init(2)=0; end
convBlock = dagnn.Conv('size', [3,3,opts.filterSize,1], 'hasBias', true, 'init', init, 'pad', 1);
net.addLayer(['conv',num2str(opts.depth+1)], convBlock, {['x',num2str(x)]}, {'prediction'}, {['filters',num2str(opts.depth+1)], ['biases',num2str(opts.depth+1)]});

net.addLayer('objective', dagnn.EuclidLoss(), ...
             {'prediction','label'}, 'objective') ;
