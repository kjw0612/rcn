function RCN_(datasetName, SF, model, outRoute)

if isempty(model)
    modelPath = ['data/exp/exp_S',num2str(SF),'_resid1_depth10/best.mat'];
else
    modelPath = ['methods/RCN/',model];
    gpu = 1;
end

load(modelPath);
net = dagnn.DagNN.loadobj(net) ;
if gpu
    net.move('gpu');
end

managableMax = 630*630;

dataDir = fullfile('data', datasetName);
f_lst = dir(fullfile(dataDir, '*.*'));
timetable = zeros(numel(f_lst),1);
for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
    if f_info.isdir, continue; end
    [~,imgName,~] = fileparts(f_lst(f_iter).name);
    im = imread(fullfile(dataDir, f_info.name));
    if size(im,3)>1
        im = rgb2ycbcr(im);
        im = im(:,:,1);
    end
    imhigh = modcrop(im, SF);
    imhigh = single(imhigh)/255;
    imlow = imresize(imhigh, 1/SF, 'bicubic');
    imlow = imresize(imlow, size(imhigh), 'bicubic');
    imlow = max(16.0/255, min(235.0/255, imlow));
    tic;
    if size(imlow,1)*size(imlow,2) > managableMax
        impred = runPatchRCN(net, imlow, gpu, 15);
    else
        if gpu, imlow = gpuArray(imlow); end;
        impred = runRCN(net, imlow, gpu);
    end
    timetable(f_iter,1) = toc;
    imwrite(impred, fullfile(outRoute, [imgName, '.png']));
end

save(fullfile(outRoute,'elapsed_time.mat'),'timetable');