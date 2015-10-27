function impred = runRCN(model, imlow, gpu)
load(model);
net = dagnn.DagNN.loadobj(net) ;
if gpu
    net.move('gpu');
    imlow = gpuArray(imlow);
end
inputs = {'input', imlow};
%net.conserveMemory = false;
net.mode = 'test';
net.eval(inputs);
pInd = getVarIndex(net,'prediction');
impred = net.vars(pInd).value;
impred = imlow+impred;
if gpu, impred = gather(impred); end


