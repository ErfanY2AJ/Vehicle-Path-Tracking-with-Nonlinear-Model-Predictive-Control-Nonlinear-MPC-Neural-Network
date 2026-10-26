function comparison = q2_compare_pair(opt,net,folder,caseName,massFactor,reverseOrder)
%Q2_COMPARE_PAIR Identical plant/reference/cost/bounds/solver for both methods.
cfg=opt.controller; cfg.Tsim=opt.simDuration; cfg.alpha=0.3;
cfg.actualMassFactor=massFactor;
cfg.progressEvery=50; cfg.net=net;
cfg.outputRoot=folder; cfg.showFigures=opt.showFigures;
assert(abs(cfg.dt-net.dt)<1e-12);
assert(norm(cfg.inputScale-net.inputScale,inf)<1e-12, ...
    'Input scaling differs from the neural training experiment.');
assert(isequal(cfg.p,net.trainingParameters), ...
    'Keep nominal predictor parameters equal to those used for training.');
caseFolder=fullfile(folder,caseName); mkdir(caseFolder);
% Warm up BOTH predictor/gradient paths with the same one-step horizon.
for method={'math','neural'}
    c=cfg; c.predictorType=method{1}; c.Np=1; c.Tsim=c.dt; c.progressEvery=0;
    quiet=opt; quiet.showFigures=false; quiet.liveEvery=0;
    q2_simulate(c,quiet);
end
order={'math','neural'}; if reverseOrder, order=fliplr(order); end
for j=1:2
    c=cfg; c.predictorType=order{j};
    fprintf('\n  %s | %s predictor | actual mass factor %.3f\n',caseName,order{j},massFactor);
    r=q2_simulate(c,opt);
    subfolder=fullfile(caseFolder,order{j}); mkdir(subfolder);
    save(fullfile(subfolder,'result.mat'),'r','-v7');
    export_results(r,subfolder);
    result.(order{j})=r;
end
summary=table;
for method={'math','neural'}
    r=result.(method{1});
    for window={'full','after5s'}
        m=r.metrics.(window{1});
        row=table(string(caseName),string(method{1}),string(window{1}),r.completed, ...
            m.positionRMSE_m,m.positionMAE_m,m.speedRMSE_mps,m.speedMAE_mps,m.Eu,m.EdeltaU, ...
            m.meanSolveTime_s,m.maxSolveTime_s,m.converged,m.feasibleNotConverged,m.fallback, ...
            r.metrics.boundViolations,r.featureOutsideTrainFraction, ...
            'VariableNames',{'Case','Predictor','Window','Completed','PositionRMSE_m', ...
            'PositionMAE_m','SpeedRMSE_mps','SpeedMAE_mps','Eu','EdeltaU', ...
            'MeanSolve_s','MaxSolve_s','Converged','FeasibleNotConverged','Fallback', ...
            'BoundViolations','ActualFeatureOutsideTrainingFraction'});
        if isempty(summary), summary=row; else, summary=[summary;row]; end %#ok<AGROW>
    end
end
disp(summary); writetable(summary,fullfile(caseFolder,'comparison.csv'));
q2_plot_comparison(result.math,result.neural,opt,caseFolder,caseName);
comparison=struct('math',result.math,'neural',result.neural,'summary',summary,'caseFolder',caseFolder);
save(fullfile(caseFolder,'paired_results.mat'),'comparison','-v7');
end
