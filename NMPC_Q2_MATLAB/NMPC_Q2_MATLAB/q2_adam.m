function [P,state] = q2_adam(P,G,state,step,rate,opt)
%Q2_ADAM Bias-corrected Adam, epsilon OUTSIDE sqrt(second moment).
fields=fieldnames(P);
if isempty(state)
    for j=1:numel(fields)
        n=fields{j}; state.m.(n)=zeros(size(P.(n))); state.v.(n)=zeros(size(P.(n)));
    end
end
normSquared=0;
for j=1:numel(fields), n=fields{j}; a=G.(n); normSquared=normSquared+sum(a(:).^2); end
if ~isfinite(normSquared), error('Training gradient became nonfinite.'); end
factor=min(1,opt.gradientClip/max(sqrt(normSquared),realmin));
for j=1:numel(fields)
    n=fields{j}; gradient=factor*G.(n);
    state.m.(n)=opt.beta1*state.m.(n)+(1-opt.beta1)*gradient;
    state.v.(n)=opt.beta2*state.v.(n)+(1-opt.beta2)*gradient.^2;
    mhat=state.m.(n)/(1-opt.beta1^step);
    vhat=state.v.(n)/(1-opt.beta2^step);
    P.(n)=P.(n)-rate*mhat./(sqrt(vhat)+opt.adamEpsilon);
end
end
