%% Experiment Framework
% Abbreviations: DO for dropout, FS for filtersize, D for Depth
run(fullfile(fileparts(mfilename('fullpath')),...
  'snudeep', 'matlab', 'vl_setupnn.m')) ;

net = {};
info = {};
exp_name = {};
pb = pbNotify('CV0o5gsA6RvL9fqixAloIVcnHKvljt4C'); % usage : pbNotify('accessToken');
% 
[net{end+1}, info{end+1}] = rcn_dag('augment', true);
exp_name{end+1} = sprintf('augment true');
[net{end+1}, info{end+1}] = rcn_dag('augment', false);
exp_name{end+1} = sprintf('augment false');
pb.notify(sprintf('Experiment %s Done', exp_name{end}));

% [net{end+1}, info{end+1}] = rcn_dag('deep_supervise', false);
% exp_name{end+1} = sprintf('deep_supervise false');
% [net{end+1}, info{end+1}] = rcn_dag('deep_supervise', true);
% exp_name{end+1} = sprintf('deep_supervise true');
% pb.notify(sprintf('Experiment Deep Supervision Done'));

% for i = 4:9
%     [net{end+1}, info{end+1}] = rcn_dag('filterSize', 2^i, 'depth', 10);
%     exp_name{end+1} = sprintf('filterSize %d (D10)', 2^i);
%     pb.notify(sprintf('Experiment %s Done', exp_name{end}));
% end
for i = 10:10:30
  [net{end+1}, info{end+1}] = rcn_dag('depth', i, 'filterSize', 256);
  exp_name{end+1} = sprintf('depth %d (FS 256)', i);
  pb.notify(sprintf('Experiment %s Done', exp_name{end}));
end
% for i = 10:10:50
%   [net{end+1}, info{end+1}] = rcn_dag('depth', i, 'filterSize', 128);
%   exp_name{end+1} = sprintf('depth %d (FS 128)', i);
%   pb.notify(sprintf('Experiment %s Done', exp_name{end}));
% end
% for i = 4:9
%     [net{end+1}, info{end+1}] = rcn_dag('dropout', 1, 'filterSize', 2^i);
%     exp_name{end+1} = sprintf('depth 10 fs %d drop 1', 2^i);
%     pb.notify(sprintf('Experiment %s Done', exp_name{end}));
% end
% for i = 8:2:20
%     [net{end+1}, info{end+1}] = rcn_dag('depth', i, 'dropout', 1, 'gpus', [1], 'filterSize', 128);
%     exp_name{end+1} = sprintf('D %d DO 1 FS 128', i);
%     pb.notify(spritf('Experiment %s Done', exp_name{end}));
% end
% [net{end+1}, info{end+1}] = rcn_dag('useBnorm', false, 'gpus', [1], 'filterSize', 32, 'dropout', 1);
% exp_name{end+1} = sprintf('no bnorm', i);
% [net{end+1}, info{end+1}] = rcn_dag('useBnorm', true,'gpus', [1], 'filterSize', 32, 'dropout', 1);
% exp_name{end+1} = sprintf('bnorm', i);
% [net{end+1}, info{end+1}] = rcn_dag('dropout', 1, 'momentum', 0.9);
% exp_name{end+1} = sprintf('momentum 0.99 DO', i);
% [net{end+1}, info{end+1}] = rcn_dag('dropout', 1, 'momentum', 0.9);
% exp_name{end+1} = sprintf('momentum 0.9 DO', i);

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