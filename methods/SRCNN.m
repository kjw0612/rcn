function SRCNN(datasetName, SF, model, outRoute)

if isempty(model)
    model = ['methods\SRCNN\9-5-5(ImageNet)\x',num2str(SF),'.mat'];
else
    model = ['methods\SRCNN\9-5-5(ImageNet)\x',num2str(model),'.mat'];
end

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
    tic;
    imlow = imresize(imlow, size(imhigh), 'bicubic');
    imlow = max(16.0/255, min(235.0/255, imlow));
    impred = runSRCNN(model, imlow);
    timetable(f_iter,1) = toc;
    imwrite(impred, fullfile(outRoute, [imgName, '.png']));
end

save(fullfile(outRoute,'elapsed_time.mat'),'timetable');