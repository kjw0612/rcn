%% Experiment Framework
% Abbreviations: DO for dropout, FS for filtersize, D for Depth
run(fullfile(fileparts(mfilename('fullpath')),...
  'snudeep', 'matlab', 'vl_setupnn.m')) ;

net = {};
info = {};
exp_name = {};
pb = pbNotify('CV0o5gsA6RvL9fqixAloIVcnHKvljt4C'); % usage : pbNotify('accessToken');
% for i = 1:9
%     [net{end+1}, info{end+1}] = rcn_dag('filterSize', 2^i);
%     exp_name{end+1} = sprintf('filterSize %d (no dropout)', 2^i);
% end
% for i = 2:2:20
%     [net{end+1}, info{end+1}] = rcn_dag('depth', i);
%     exp_name{end+1} = sprintf('depth %d (no dropout)', i);
% end
% for i = 4:9
%     [net{end+1}, info{end+1}] = rcn_dag('dropout', 1, 'filterSize', 2^i);
%     exp_name{end+1} = sprintf('depth 10 fs %d drop 1', 2^i);
%     pb.notify(sprintf('Experiment %s Done', exp_name{end}));
% end
for i = 8:2:20
    [net{end+1}, info{end+1}] = rcn_dag('depth', i, 'dropout', 1, 'gpus', [1], 'filterSize', 128);
    exp_name{end+1} = sprintf('D %d DO 1 FS 128', i);
    pb.notify(spritf('Experiment %s Done', exp_name{end}));
end
pb.notify('RCN Experiment Done');

%%
figure(1) ; clf ;
subplot(1,2,1) ;
for i = 1:numel(net)
    plot([info{i}.val.objective]) ; hold on ;
end
xlabel('Training samples [x10^3]'); ylabel('objective (val)') ;
grid on ;
h=legend(exp_name) ;
title('objective') ;

subplot(1,2,2);

for i =1:numel(net)
    plot(info{i}.test) ; hold on ;
    exp_name{i} = sprintf('%s max PSNR :%f', exp_name{i}, max(info{i}.test));
end
h=legend(exp_name, 'location', 'southeast') ;
grid on ;
xlabel('Training samples [x10^3]'); ylabel('error') ;
title('PSNR') ;
drawnow ;
save('exp_depth_dropout.mat');