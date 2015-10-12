function impred = runRCN(model, imlow)
load(model);
inputs = {'input', imlow};
net = dagnn.DagNN.loadobj(net) ;
%net.conserveMemory = false;
net.eval(inputs);
pInd = getVarIndex(net,'prediction');
impred = net.vars(pInd).value;
impred = imlow+impred;

