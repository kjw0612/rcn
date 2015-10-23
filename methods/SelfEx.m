function SelfEx(datasetName, SF, outRoute)
% =========================================================================
% Start super-resolving images
% =========================================================================
startup;
% Initialize the paramters for super-resolution
opt = sr_init_opt(SF);

% Process all images in the dataset
dataDir = fullfile('data', datasetName);
lowdataDir = fullfile('data','forSelfEx', [datasetName,'_LR_x',num2str(SF)]);
if ~exist(lowdataDir,'dir')
    mkdir(lowdataDir);
end
f_lst = dir(fullfile(dataDir, '*.*'));
timetable = zeros(numel(f_lst),1);

for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
    if f_info.isdir, continue; end
    [~,imgName,~] = fileparts(f_lst(f_iter).name);
    im = imread(fullfile(dataDir, f_info.name));
    imhigh = modcrop(im, SF);
    imhigh = single(imhigh)/255;
    imlow = imresize(imhigh, 1/SF, 'bicubic');
    imwrite(imlow, fullfile(lowdataDir,[imgName,'.png']));
    
    filePath = [];
    filePath.dataPath    = lowdataDir;
    filePath.imgFileName = [imgName,'.png'];
    tic;
    impred = runSelfEx(filePath, opt);
    timetable(f_iter,1) = toc;    
    % Save results
    imwrite(impred, fullfile(outRoute, [imgName, '.png']));
end
       
save(fullfile(outRoute,'elapsed_time.mat'),'timetable');