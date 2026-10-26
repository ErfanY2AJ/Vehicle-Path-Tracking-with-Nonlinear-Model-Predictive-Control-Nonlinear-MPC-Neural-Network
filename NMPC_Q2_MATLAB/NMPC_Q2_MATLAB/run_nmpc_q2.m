function report = run_nmpc_q2(mode,overrides)
%RUN_NMPC_Q2 Step-by-step Question 2, with live figures and saved checkpoints.
% report=run_nmpc_q2('quick');
% report=run_nmpc_q2('standard',struct('pauseBetweenSteps',true));
if nargin<1, mode='quick'; end
if nargin<2, overrides=struct; end
here=fileparts(mfilename('fullpath')); addpath(here,fullfile(here,'core'));
opt=q2_config(mode); fields=fieldnames(overrides);
for j=1:numel(fields)
    if ~isfield(opt,fields{j}), error('Unknown Question 2 option: %s',fields{j}); end
    opt.(fields{j})=overrides.(fields{j});
end
assert(opt.numRuns>=40 && opt.numRuns==round(opt.numRuns));
assert(opt.stepsPerRun>=200 && opt.stepsPerRun==round(opt.stepsPerRun));
assert(opt.maxEpochs>=1 && opt.maxEpochs==round(opt.maxEpochs));
assert(opt.batchSize>=1 && opt.batchSize==round(opt.batchSize));
assert(opt.simDuration>5 && opt.controller.dt>0);
assert(opt.trainFraction>0 && opt.validationFraction>0 && opt.trainFraction+opt.validationFraction<1);
assert(numel(opt.hidden)==2 && all(opt.hidden>=1) && all(opt.hidden==round(opt.hidden)));
assert(opt.learningRate>0 && opt.patience>=1 && opt.printEvery>=1);
assert(all(opt.horizons>=1) && all(opt.horizons==round(opt.horizons)) ...
    && max(opt.horizons)<=opt.stepsPerRun);
assert(opt.beta1>=0 && opt.beta1<1 && opt.beta2>=0 && opt.beta2<1);
assert(opt.adamEpsilon>0 && opt.gradientClip>0 && opt.massFactor>0);
assert(opt.controller.Np>=1 && opt.controller.Np==round(opt.controller.Np));
assert(opt.maxAttemptsPerRun>=1 && opt.maxAttemptsPerRun==round(opt.maxAttemptsPerRun));
assert(opt.controlHoldSteps>=1 && opt.controlHoldSteps==round(opt.controlHoldSteps));
if ~exist(opt.outputRoot,'dir'), mkdir(opt.outputRoot); end
prefix=fullfile(opt.outputRoot,['q2_' opt.mode '_' datestr(now,'yyyymmdd_HHMMSS')]);
folder=prefix; suffix=1;
while exist(folder,'dir'), folder=sprintf('%s_%d',prefix,suffix); suffix=suffix+1; end
mkdir(folder);
fprintf('Question 2 | %s mode | output: %s\n',opt.mode,folder);
fprintf('Reference: %s\n',opt.controller.referenceLabel);
fprintf('Both controllers: dt=%g, Np=%d, solver=%s, gradTol=%g, stepTol=%g\n', ...
    opt.controller.dt,opt.controller.Np,opt.controller.solver, ...
    opt.controller.gradTol,opt.controller.stepTol);
save(fullfile(folder,'step01_options.mat'),'opt');
q2_stage(1,'Verify backpropagation, learned Jacobians, and the common objective',opt);
q2_self_tests();
q2_stage(2,'Generate independent nominal vehicle runs',opt);
data=q2_generate_data(opt,folder);
q2_stage(3,'Split whole runs, fit training-only normalization, inspect coverage',opt);
split=q2_prepare_data(data,opt,folder);
q2_stage(4,'Train the neural dynamics and select the validation epoch',opt);
[net,history]=q2_train_network(split,opt,folder);
q2_stage(5,'Evaluate one-step and recursive prediction on unseen test runs',opt);
evaluation=q2_evaluate_network(data,split,net,opt,folder);
q2_stage(6,'Compare mathematical and neural NMPC on the nominal mild path',opt);
nominal=q2_compare_pair(opt,net,folder,'nominal_mild',1,false);
q2_stage(7,sprintf('Repeat with actual plant mass factor %.3f',opt.massFactor),opt);
uncertaintyName='mass_plus10pct';
if abs(opt.massFactor-1.1)>1e-12
    uncertaintyName=sprintf('mass_factor_%g',opt.massFactor);
end
uncertain=q2_compare_pair(opt,net,folder,uncertaintyName,opt.massFactor,true);
q2_stage(8,'Display and save the final comparison',opt);
summary=[nominal.summary;uncertain.summary];
disp(summary); writetable(summary,fullfile(folder,'final_comparison.csv'));
report=struct('folder',folder,'options',opt,'history',history, ...
    'evaluation',evaluation,'nominal',nominal,'uncertainty',uncertain,'summary',summary);
save(fullfile(folder,'q2_report.mat'),'report','net','-v7');
fid=fopen(fullfile(folder,'experiment_notes.txt'),'w');
if fid<0, error('Cannot write experiment notes.'); end
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'MATLAB: %s\nMode: %s\nData seed: %d\n',version,opt.mode,opt.seed);
fprintf(fid,'All training data: exact nominal vehicle_step, no closed-loop path data.\n');
fprintf(fid,'Whole-run splitting; all fitted transformations use training runs only.\n');
fprintf(fid,'Predictor: learned body-frame increments, affine skip plus tanh MLP.\n');
fprintf(fid,'Both NMPC controllers use analytical Jacobians and the same adjoint cost gradient.\n');
fprintf(fid,'Plant, objective, horizon, bounds, warm start, and fallback are identical.\n');
fprintf(fid,'Timing excludes live plotting, plant update, and CSV/MAT export.\n');
fprintf(fid,'Practice reference replaces the unavailable original assignment path.\n');
fprintf(fid,'Training and test metrics are separate from closed-loop controller metrics.\n');
fprintf('Experiment pipeline finished. Inspect Completed flags in report.summary.\n');
fprintf('Inspect report.evaluation.recursive for held-out prediction errors.\n');
fprintf('Saved all stage results in: %s\n',folder);
end
