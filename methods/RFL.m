function RFL(datasetName, SF, model, outRoute)

if isempty(model)
    modelPath = ['methods/RFL/srf_sf-',num2str(SF),'_T-10_ARF-0.mat'];
else
    modelPath = ['methods/RFL/',model];
end

srforest = srForestLoad(modelPath);

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
    impred = runRFL(srforest, imlow);
    timetable(f_iter,1) = toc;
    imwrite(impred, fullfile(outRoute, [imgName, '.png']));
end

save(fullfile(outRoute,'elapsed_time.mat'),'timetable');