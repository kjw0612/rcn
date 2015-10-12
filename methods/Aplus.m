function Aplus(datasetName, SF, model, outRoute)

Aplus_PPs = []; dictSize = 1024; clusterSize = 2048;
if dictSize < 1024
    lambda = 0.01;
elseif dictSize < 2048
    lambda = 0.1;
elseif dictSize < 8192
    lambda = 1;
else
    lambda = 5;
end

tic;
if isempty(model)
    mat_file = ['methods\Aplus\conf_Zeyde_' num2str(dictSize) '_finalx' num2str(SF) '.mat'] ;
    if exist(mat_file,'file')
        load(mat_file, 'conf');
    else
        error([mat_file ' : Zeyde model is needed']);
    end
    fname = ['methods\Aplus\Aplus_x' num2str(SF) '_' num2str(dictSize) 'atoms' num2str(clusterSize) 'nn_5mil.mat'];
    if exist(fname,'file')
        load(fname);
    else
        error([fname ' : A+ model is needed']);
    end
else
    mat_file = ['methods\Aplus\conf_Zeyde_' num2str(dictSize) '_finalx' num2str(model) '.mat'];
    if exist(mat_file,'file')
        load(mat_file, 'conf');
    else
        error([mat_file ' : Zeyde model is needed']);
    end
    fname = ['methods\Aplus\Aplus_x' num2str(model) '_' num2str(dictSize) 'atoms' num2str(clusterSize) 'nn_5mil.mat'];
    if exist(fname,'file')
        load(fname);
    else
        error([fname ' : A+ model is needed']);
    end
end
basetime = toc;


conf.PPs = Aplus_PPs;
conf.ProjM = inv(conf.dict_lores'*conf.dict_lores+lambda*eye(size(conf.dict_lores,2)))*conf.dict_lores';
conf.PP = (1+lambda)*conf.dict_hires*conf.ProjM;

conf.points = [1:1:size(conf.dict_lores,2)];
conf.pointslo = conf.dict_lores(:,conf.points);
conf.pointsloPCA = conf.pointslo'*conf.V_pca';


dataDir = fullfile('data', datasetName);
f_lst = dir(fullfile(dataDir, '*.*'));
timetable = zeros(numel(f_lst),1);
for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
    if f_info.isdir, continue; end
    [~,imgName,~] = fileparts(f_lst(f_iter).name);
    im = imread(fullfile(dataDir, f_info.name));
    im = rgb2ycbcr(im);
    im = im(:,:,1);
    imhigh = modcrop(im, SF);
    imhigh = single(imhigh)/255;
    imlow = imresize(imhigh, 1/SF, 'bicubic');
    tic;
    %imlow = imresize(imlow, size(imhigh), 'bicubic');
    %imlow = max(16.0/255, min(235.0/255, imlow));
    impred = runAplus(conf, {imlow});     
    timetable(f_iter,1) = toc + basetime;
    imwrite(impred{1}, fullfile(outRoute, [imgName, '.png']));
end

save(fullfile(outRoute,'elapsed_time.mat'),'timetable');