function opt = q2_config(mode)
%Q2_CONFIG Base-MATLAB neural NMPC experiment; all calculations use doubles.
if nargin<1, mode='quick'; end
here=fileparts(mfilename('fullpath')); addpath(fullfile(here,'core'));
opt.mode=lower(char(mode));
opt.seed=42;
opt.numRuns=40;
opt.stepsPerRun=200;
opt.maxAttemptsPerRun=100;
opt.controlHoldSteps=8;
opt.trainFraction=0.70;
opt.validationFraction=0.15;
opt.hidden=[32 32];
opt.maxEpochs=150;
opt.batchSize=512;
opt.learningRate=2e-3;
opt.beta1=0.9;
opt.beta2=0.999;
opt.adamEpsilon=1e-8;
opt.weightDecay=1e-6;
opt.gradientClip=5;
opt.patience=45;
opt.minImprovement=1e-7;
opt.printEvery=10;
opt.simDuration=15;
opt.horizons=[1 5 10 20];
opt.massFactor=1.10;
opt.showFigures=true;
opt.saveFigures=true;
opt.pauseBetweenSteps=false;
opt.liveEvery=10;
opt.outputRoot=fullfile(here,'results_q2');
opt.controller=nmpc_config();
opt.controller.outputRoot=opt.outputRoot;
switch opt.mode
    case 'quick'
    case 'standard'
        opt.numRuns=60; opt.stepsPerRun=250;
        opt.maxEpochs=400; opt.patience=70; opt.simDuration=100;
    otherwise
        error('Mode must be quick or standard.');
end
end
