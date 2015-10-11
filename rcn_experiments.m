%% Experiment Framework

net = {};
info = {};
exp_name = {};
for i = 1:9
    [net{end+1}, info{end+1}] = rcn_dag('filterSize', 2^i);
    exp_name{end+1} = sprintf('filterSize %d (no dropout)', 2^i);
end

%%
figure(1) ; clf ;
subplot(1,2,1) ;
for i = 1:numel(net)
    val = zeros(1,numel(info{1}.val));
    val(:) = info{i}.val.objective;
    plot(val) ; hold on ;
end
xlabel('Training samples [x10^3]'); ylabel('objective (val)') ;
grid on ;
h=legend(exp_name) ;
title('objective') ;

subplot(1,2,2);

for i =1:numel(net)
    plot(info{i}.test) ; hold on ;
end
h=legend(exp_name, 'location', 'southeast') ;
grid on ;
xlabel('Training samples [x10^3]'); ylabel('error') ;
title('PSNR') ;
drawnow ;