function evaluation = q2_evaluate_network(data,split,net,opt,folder)
%Q2_EVALUATE_NETWORK One-step and truly recursive held-out-run evaluation.
a=split.test; predicted=q2_nn_step(a.X,a.U,net);
error=predicted-a.Xnext; persistence=a.X-a.Xnext;
actualDelta=a.Xnext-a.X; estimatedDelta=predicted-a.X;
sst=sum((actualDelta-mean(actualDelta,2)).^2,2);
r2=1-sum((estimatedDelta-actualDelta).^2,2)./sst;
r2(sst<1e-20)=NaN;
oneStep=table(["px";"py";"v";"phi";"T"],["m";"m";"m/s";"rad";"N m"], ...
    sqrt(mean(error.^2,2)),mean(abs(error),2), ...
    sqrt(mean(persistence.^2,2)),mean(abs(persistence),2),r2, ...
    'VariableNames',{'State','Unit','NeuralRMSE','NeuralMAE', ...
    'PersistenceRMSE','PersistenceMAE','IncrementR2'});
fprintf('\n  One-step test-set metrics (next-state errors, increment R2):\n'); disp(oneStep);
writetable(oneStep,fullfile(folder,'one_step_test_metrics.csv'));
recursive=table; horizons=opt.horizons(:).';
for H=horizons
    pnn=[]; vnn=[]; pp=[]; vp=[]; bad=0; nonpositive=0;
    for id=split.runIDs{3}
        run=data.runs{id}; K=size(run.U,2);
        starts=1:K-H+1; X=run.X(:,starts); initial=X;
        for j=1:H
            % Never replace predicted intermediate X with recorded states.
            X=q2_nn_step(X,run.U(:,starts+j-1),net);
        end
        truth=run.X(:,starts+H);
        pnn=[pnn,sqrt(sum((X(1:2,:)-truth(1:2,:)).^2,1))]; %#ok<AGROW>
        vnn=[vnn,abs(X(3,:)-truth(3,:))]; %#ok<AGROW>
        pp=[pp,sqrt(sum((initial(1:2,:)-truth(1:2,:)).^2,1))]; %#ok<AGROW>
        vp=[vp,abs(initial(3,:)-truth(3,:))]; %#ok<AGROW>
        bad=bad+nnz(any(~isfinite(X),1)); nonpositive=nonpositive+nnz(X(3,:)<=0);
    end
    pmean=mean(pnn); vmean=mean(vnn);
    if bad>0, pmean=Inf; vmean=Inf; end
    row=table(H,numel(pnn),pmean,vmean,mean(pp),mean(vp),bad,nonpositive, ...
        'VariableNames',{'HorizonSteps','StartingPoints','NeuralMeanPositionError_m', ...
        'NeuralMeanAbsSpeedError_mps','PersistenceMeanPositionError_m', ...
        'PersistenceMeanAbsSpeedError_mps','NonfiniteEndpoints','NonpositiveSpeedEndpoints'});
    if isempty(recursive), recursive=row; else, recursive=[recursive;row]; end %#ok<AGROW>
end
fprintf('  Recursive errors: every possible start in each held-out test run.\n'); disp(recursive);
writetable(recursive,fullfile(folder,'recursive_test_metrics.csv'));
f=q2_figure('Step 5 - held-out prediction quality',opt);
tiledlayout(2,2,'TileSpacing','compact');
nexttile; sample=1:min(1000,size(error,2));
plot(sample,error(3,sample)); grid on; xlabel('Held-out transition'); ylabel('One-step speed error (m/s)');
nexttile; plot(sample,error(4,sample)); grid on;
xlabel('Held-out transition'); ylabel('One-step heading error (rad)');
nexttile; semilogy(horizons,max(recursive.NeuralMeanPositionError_m,realmin),'bo-', ...
    horizons,max(recursive.PersistenceMeanPositionError_m,realmin),'ks--');
grid on; xlabel('Horizon (steps)'); ylabel('Mean position error (m)'); legend('Neural','Persistence');
nexttile; semilogy(horizons,max(recursive.NeuralMeanAbsSpeedError_mps,realmin),'bo-', ...
    horizons,max(recursive.PersistenceMeanAbsSpeedError_mps,realmin),'ks--');
grid on; xlabel('Horizon (steps)'); ylabel('Mean absolute speed error (m/s)');
q2_save_figure(f,'step05_prediction_errors',opt,folder);
run=data.runs{split.runIDs{3}(1)}; H=max(horizons);
roll=zeros(5,H+1); roll(:,1)=run.X(:,1);
for j=1:H, roll(:,j+1)=q2_nn_step(roll(:,j),run.U(:,j),net); end
f=q2_figure('Step 5 - recursive example, no state replacement',opt);
tiledlayout(1,3,'TileSpacing','compact'); t=(0:H)*data.dt;
nexttile; plot(run.X(1,1:H+1),run.X(2,1:H+1),'k-o',roll(1,:),roll(2,:),'r--');
axis equal; grid on; xlabel('x (m)'); ylabel('y (m)'); legend('Recorded truth','Recursive neural');
nexttile; plot(t,run.X(3,1:H+1),'k-o',t,roll(3,:),'r--'); grid on;
xlabel('Time (s)'); ylabel('Speed (m/s)');
nexttile; plot(t,roll(4,:)-run.X(4,1:H+1)); grid on;
xlabel('Time (s)'); ylabel('Unwrapped heading error (rad)');
q2_save_figure(f,'step05_recursive_example',opt,folder);
evaluation=struct('oneStep',oneStep,'recursive',recursive,'testRunIDs',split.runIDs{3}, ...
    'startingPointRule','All starts 1:K-H+1 independently in every held-out run');
save(fullfile(folder,'step05_prediction_evaluation.mat'),'evaluation','-v7');
end
