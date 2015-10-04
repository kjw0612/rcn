%% Batch normalization effect experiment

[net_bn, info_bn] = rcn(...
  'useBnorm', true);

[net_fc, info_fc] = rcn(...
  'useBnorm', false);

%%
figure(1) ; clf ;
subplot(1,2,1) ;
semilogy(info_fc.val.objective, 'k') ; hold on ;
semilogy(info_bn.val.objective, 'b') ;
xlabel('Training samples [x10^3]'); ylabel('energy') ;
grid on ;
h=legend('BASE', 'BNORM') ;
set(h,'color','none');
title('objective') ;

subplot(1,2,2) ;
nProblem = numel(info_fc.test.error);
base = info_fc.test.error{1}.ours;
bnorm = info_bn.test.error{1}.ours;
for problem_iter = 2:nProblem
    base = base + info_fc.test.error{problem_iter}.ours;
    bnorm = bnorm + info_bn.test.error{problem_iter}.ours;
end
base = base / nProblem;
bnorm = bnorm / nProblem;

plot(base, 'k') ; hold on ; % first row for top1e
plot(bnorm, 'b') ;
h=legend('BASE','BNORM') ;
grid on ;
xlabel('Training samples [x10^3]'); ylabel('error') ;
set(h,'color','none') ;
title('error') ;
drawnow ;