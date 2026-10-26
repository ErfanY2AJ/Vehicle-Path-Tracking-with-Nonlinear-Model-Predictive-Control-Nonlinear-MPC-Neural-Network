function split = q2_prepare_data(data,opt,folder)
%Q2_PREPARE_DATA Split complete runs BEFORE fitting any normalization.
R=numel(data.runs);
old=rng; cleanup=onCleanup(@()rng(old)); %#ok<NASGU>
rng(opt.seed+1); order=randperm(R);
nTrain=floor(opt.trainFraction*R);
nVal=floor(opt.validationFraction*R);
assert(nTrain>0 && nVal>0 && R-nTrain-nVal>0);
ids={sort(order(1:nTrain)),sort(order(nTrain+1:nTrain+nVal)), ...
    sort(order(nTrain+nVal+1:end))};
names={'train','validation','test'};
split=struct('dt',data.dt,'inputScale',data.inputScale,'p',data.p,'runIDs',{ids});
for j=1:3
    X=[]; U=[]; Xnext=[]; runID=[]; %#ok<NASGU>
    for id=ids{j}
        a=data.runs{id}; K=size(a.U,2);
        X=[X,a.X(:,1:K)]; U=[U,a.U]; Xnext=[Xnext,a.X(:,2:K+1)]; %#ok<AGROW>
        runID=[runID,id*ones(1,K)]; %#ok<AGROW>
    end
    part=struct('X',X,'U',U,'Xnext',Xnext,'runID',runID);
    part.features=q2_features(X,U,data.inputScale);
    part.targets=q2_body_targets(X,Xnext);
    split.(names{j})=part;
end
train=split.train;
mu=mean(train.features,2); sigma=std(train.features,0,2);
sigma(sigma<1e-12)=1;
q=(train.features-mu)./sigma;
% All affine coefficients, residual scales, and constant-output decisions
% are estimated from TRAINING runs only. No model equation is used here.
affine=train.targets/[ones(1,size(q,2));q];
residual=train.targets-affine*[ones(1,size(q,2));q];
residualMean=mean(residual,2);
residualScale=std(residual,0,2);
active=residualScale>1e-10;
residualScale(~active)=0;
prep=struct('featureMean',mu,'featureStd',sigma,'affine',affine, ...
    'residualMean',residualMean,'residualScale',residualScale, ...
    'active',active,'featureMin',min(train.features,[],2), ...
    'featureMax',max(train.features,[],2));
for j=1:3
    part=split.(names{j});
    part.Q=(part.features-mu)./sigma;
    residual=part.targets-affine*[ones(1,size(part.Q,2));part.Q]-residualMean;
    safeScale=residualScale; safeScale(~active)=1;
    part.Y=residual./safeScale;
    part.Y(~active,:)=0;
    split.(names{j})=part;
end
split.prep=prep;
assert(isempty(intersect(ids{1},ids{2})) && isempty(intersect(ids{1},ids{3})) ...
    && isempty(intersect(ids{2},ids{3})));
coverage=table;
for j=1:3
    a=split.(names{j}); z=a.U./data.inputScale;
    row=table(string(names{j}),numel(ids{j}),size(a.X,2), ...
        min(a.X(3,:)),max(a.X(3,:)),min(z(1,:)),max(z(1,:)),min(z(2,:)),max(z(2,:)), ...
        'VariableNames',{'Partition','Runs','Transitions','MinSpeed','MaxSpeed', ...
        'MinTorqueNorm','MaxTorqueNorm','MinSteeringNorm','MaxSteeringNorm'});
    if j==1, coverage=row; else, coverage=[coverage;row]; end %#ok<AGROW>
end
disp(coverage);
assert(min(train.U(1,:)/data.inputScale(1))<=-1+1e-12 && ...
    max(train.U(1,:)/data.inputScale(1))>=1-1e-12);
assert(min(train.U(2,:)/data.inputScale(2))<=-1+1e-12 && ...
    max(train.U(2,:)/data.inputScale(2))>=1-1e-12);
writetable(coverage,fullfile(folder,'partition_coverage.csv'));
assignment=zeros(R,1);
for j=1:3, assignment(ids{j})=j; end
writetable(table((1:R).',assignment,'VariableNames',{'RunID','PartitionCode'}), ...
    fullfile(folder,'run_partitions.csv'));
save(fullfile(folder,'step03_partitions.mat'),'split','coverage','-v7');
f=q2_figure('Step 3 - training coverage',opt);
tiledlayout(2,3,'TileSpacing','compact');
nexttile; histogram(train.U(1,:)/data.inputScale(1),30); xlabel('Normalized torque'); grid on;
nexttile; histogram(train.U(2,:)/data.inputScale(2),30); xlabel('Normalized steering'); grid on;
nexttile; histogram(train.X(3,:),30); xlabel('Speed (m/s)'); grid on;
nexttile; histogram(atan2(sin(train.X(4,:)),cos(train.X(4,:))),30);
xlabel('Wrapped heading (rad)'); grid on;
nexttile; histogram(train.X(5,:),30); xlabel('Actual torque (N m)'); grid on;
nexttile; scatter(train.X(3,1:5:end),train.U(1,1:5:end)/data.inputScale(1),5,'.');
xlabel('Speed (m/s)'); ylabel('Normalized torque'); grid on;
sgtitle('Training runs only - no transition-level mixing');
q2_save_figure(f,'step03_training_coverage',opt,folder);
fprintf('  Normalization and affine fit use %d training runs only.\n',nTrain);
end
