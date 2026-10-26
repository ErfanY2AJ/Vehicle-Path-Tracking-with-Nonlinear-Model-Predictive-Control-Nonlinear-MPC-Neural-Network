function q2_self_tests()
%Q2_SELF_TESTS Nontrivial derivative/indexing tests; no training toolbox.
here=fileparts(mfilename('fullpath')); addpath(here,fullfile(here,'core'));
old=rng; cleanup=onCleanup(@()rng(old)); %#ok<NASGU>
rng(710); cfg=nmpc_config();
active=true(5,1); P=q2_mlp_init([3 2],active);
Q=randn(4,7); Y=randn(5,7); decay=1e-4;
[~,G]=q2_mlp_loss(P,Q,Y,active,decay);
fields=fieldnames(P); num=0; den=0; h=1e-6;
for j=1:numel(fields)
    n=fields{j};
    for i=1:numel(P.(n))
        plus=P; minus=P;
        plus.(n)(i)=plus.(n)(i)+h; minus.(n)(i)=minus.(n)(i)-h;
        fd=(q2_mlp_loss(plus,Q,Y,active,decay)-q2_mlp_loss(minus,Q,Y,active,decay))/(2*h);
        exact=G.(n)(i); num=num+(fd-exact)^2; den=den+fd^2;
    end
end
backpropError=sqrt(num)/max(1,sqrt(den));
assert(backpropError<1e-6,'MLP backpropagation check failed');
net=struct('params',P,'inputScale',cfg.inputScale,'dt',cfg.dt);
net.prep=struct('featureMean',[15;0;0;0],'featureStd',[5;.5;1;1], ...
    'affine',0.01*randn(5,5),'residualMean',zeros(5,1), ...
    'residualScale',[.03;.02;.1;.02;2]);
x=[2;-3;14;.4;120]; u=[300;.12];
[xn,A,B]=q2_nn_step(x,u,net); An=zeros(5); Bn=zeros(5,2); h=1e-5;
for j=1:5
    d=zeros(5,1); d(j)=h;
    An(:,j)=(q2_nn_step(x+d,u,net)-q2_nn_step(x-d,u,net))/(2*h);
end
for j=1:2
    d=zeros(2,1); d(j)=h;
    Bn(:,j)=(q2_nn_step(x,u+d,net)-q2_nn_step(x,u-d,net))/(2*h);
end
stateError=norm(A-An,'fro')/max(1,norm(An,'fro'));
inputError=norm(B-Bn,'fro')/max(1,norm(Bn,'fro'));
fprintf('  Learned Jacobian relative errors: A=%.6g, B=%.6g (limit 1e-6).\n', ...
    stateError,inputError);
assert(stateError<1e-6 && inputError<1e-6, ...
    sprintf('Learned-dynamics Jacobian failed: A=%.6g, B=%.6g',stateError,inputError));
shift=[1000;-800;0;2*pi;0];
assert(norm(q2_nn_step(x+shift,u,net)-xn-shift,inf)<1e-9, ...
    'Translation/heading-periodicity reconstruction failed');
cfg.Np=3; cfg.Tsim=1; ref=make_reference(cfg);
yr=ref.Y(:,1:cfg.Np+1);
zr=[equilibrium_torque(yr(3,2:end),cfg.p)/cfg.inputScale(1);zeros(1,cfg.Np)];
z=0.1*randn(2*cfg.Np,1); prev=[.08;.03];
cfg.predictorType='math'; cfg.net=net;
[ja,ga]=q2_objective(z,x,prev,yr,zr,cfg);
[jb,gb]=nmpc_objective(z,x,prev,yr,zr,cfg);
assert(abs(ja-jb)<1e-12 && norm(ga-gb,inf)<1e-12,'Mathematical baseline changed');
cfg.predictorType='neural';
fun=@(v)q2_objective(v,x,prev,yr,zr,cfg);
[~,g]=fun(z); gn=zeros(size(z)); h=1e-6;
for j=1:numel(z)
    d=zeros(size(z)); d(j)=h; gn(j)=(fun(z+d)-fun(z-d))/(2*h);
end
objectiveError=norm(g-gn)/max(1,norm(gn));
assert(objectiveError<1e-5,'Neural single-shooting objective gradient failed');
opt=q2_config(); [updated,~]=q2_adam(P,G,[],1,opt.learningRate,opt);
% First Adam step has a closed-form update, including global clipping.
total=0;
for j=1:numel(fields), a=G.(fields{j}); total=total+sum(a(:).^2); end
factor=min(1,opt.gradientClip/max(sqrt(total),realmin));
expected=P.W1-opt.learningRate*(factor*G.W1)./(abs(factor*G.W1)+opt.adamEpsilon);
assert(norm(updated.W1-expected,inf)<1e-12,'Adam bias correction failed');
fprintf('  PASS: backprop %.3g | neural A %.3g | neural B %.3g | objective %.3g\n', ...
    backpropError,stateError,inputError,objectiveError);
fprintf('  PASS: Q1 objective equivalence, translation equivariance, angular periodicity, Adam.\n');
end
