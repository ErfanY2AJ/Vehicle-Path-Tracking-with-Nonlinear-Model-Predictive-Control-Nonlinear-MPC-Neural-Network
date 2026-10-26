function data = q2_generate_data(opt,folder)
%Q2_GENERATE_DATA Independent nominal rollouts, never closed-loop test paths.
cfg=opt.controller; p=cfg.p; dt=cfg.dt; scale=cfg.inputScale;
assert(opt.numRuns>=40 && opt.stepsPerRun>=200, ...
    'The assignment requires at least 40 runs of 200 valid steps.');
old=rng; cleanup=onCleanup(@()rng(old)); %#ok<NASGU>
rng(opt.seed);
R=opt.numRuns; K=opt.stepsPerRun;
runs=cell(R,1); restarts=zeros(R,1);
for run=1:R
    valid=false;
    for attempt=1:opt.maxAttemptsPerRun
        X=zeros(5,K+1); U=zeros(2,K);
        % Stratify initial speed over 5-25 m/s without creating shared runs.
        v0=5+20*((run-1+rand)/R);
        phi0=-pi+2*pi*rand;
        T0=equilibrium_torque(v0,p)+0.6*scale(1)*(2*rand-1);
        X(:,1)=[400*rand(2,1)-200;v0;phi0;T0];
        command=zeros(2,1); valid=true;
        for k=1:K
            if mod(k-1,opt.controlHoldSteps)==0
                block=1+floor((k-1)/opt.controlHoldSteps);
                command=2*rand(2,1)-1;
                % Endpoint excitation is repeated in every independent run.
                if block==1, command=[1;1]; end
                if block==2, command=[-1;-1]; end
                if block>2
                    if rand<0.25, command(1)=2*(rand>0.5)-1; end
                    if rand<0.25, command(2)=2*(rand>0.5)-1; end
                end
            end
            % Change proposed controls, never repair/clip a resulting state.
            applied=command;
            if X(3,k)<6, applied(1)=1; end
            if X(3,k)>25, applied(1)=-1; end
            U(:,k)=scale.*applied;
            proposed=vehicle_step(X(:,k),U(:,k),p,dt);
            if any(~isfinite(proposed)) || proposed(3)<=0
                valid=false; break
            end
            X(:,k+1)=proposed;
        end
        if valid, break; end
        % Discard the ENTIRE physical run. No fragments cross partitions.
        restarts(run)=restarts(run)+1;
    end
    if ~valid
        error('Could not obtain run %d with %d consecutive valid steps.',run,K);
    end
    runs{run}=struct('id',run,'X',X,'U',U,'t',(0:K)*dt);
    if mod(run,10)==0 || run==R
        fprintf('  Accepted %d/%d independent runs (%d transitions).\n',run,R,run*K);
    end
end
data=struct('runs',{runs},'dt',dt,'p',p,'inputScale',scale, ...
    'seed',opt.seed,'restarts',restarts, ...
    'source','Exact nominal vehicle_step; independent random-control runs');
save(fullfile(folder,'step02_dataset.mat'),'data','opt','-v7');
% CSV preserves run identity and the exact x,u,x_next triplets.
M=R*K; numeric=zeros(M,14);
for run=1:R
    a=runs{run}; rows=(run-1)*K+(1:K);
    numeric(rows,:)=[run*ones(K,1),(0:K-1).',a.X(:,1:end-1).',a.U.',a.X(:,2:end).'];
end
names={'RunID','Step','px','py','v','phi','T','Tu','delta', ...
    'pxNext','pyNext','vNext','phiNext','TNext'};
writetable(array2table(numeric,'VariableNames',names),fullfile(folder,'dataset.csv'));
fprintf('  Invalid-run restarts: %d. No invalid transition was stored.\n',sum(restarts));
end
