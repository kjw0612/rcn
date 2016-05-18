function VDSR(datasetName, SF, model, outRoute)

if isempty(model)
    modelPath = ['methods/VDSR/SFSR_291plus'];
else
    modelPath = ['methods/VDSR/',model];
    gpu = 1;
end

model = load(modelPath);
net = model.net;

% managableMax = 300000;

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
    if gpu, imlow = gpuArray(imlow); end;
    %dummy run
    if f_iter ==3, runVDSR(net, imlow, gpu); end;
    tic;
    impred = runVDSR(net, imlow, gpu);    
    timetable(f_iter,1) = toc;
    imwrite(impred, fullfile(outRoute, [imgName, '.png']));
end

save(fullfile(outRoute,'elapsed_time.mat'),'timetable');