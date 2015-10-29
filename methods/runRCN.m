function impred = runRCN(net, imlow, gpu)
inputs = {'input', imlow};
%net.conserveMemory = false;
net.mode = 'test';
net.eval(inputs);
pInd = getVarIndex(net,'prediction');
impred = net.vars(pInd).value;
impred = imlow+impred;
if gpu, impred = gather(impred); end