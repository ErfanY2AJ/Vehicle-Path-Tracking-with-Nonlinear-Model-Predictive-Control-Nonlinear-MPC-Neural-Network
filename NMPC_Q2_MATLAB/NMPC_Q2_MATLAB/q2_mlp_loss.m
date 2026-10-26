function [loss,G,dataLoss] = q2_mlp_loss(P,Q,Y,active,weightDecay)
%Q2_MLP_LOSS Explicit backpropagation, no numerical training gradients.
[prediction,A1,A2]=q2_mlp_forward(P,Q);
error=(prediction-Y).*active;
denom=max(1,nnz(active))*size(Q,2);
dataLoss=sum(error(:).^2)/denom;
regularizer=0.5*weightDecay*(sum(P.W1(:).^2)+sum(P.W2(:).^2)+sum(P.W3(:).^2));
loss=dataLoss+regularizer;
if nargout>1
    d3=2*error/denom;
    G.W3=d3*A2.'+weightDecay*P.W3; G.b3=sum(d3,2);
    d2=(P.W3.'*d3).*(1-A2.^2);
    G.W2=d2*A1.'+weightDecay*P.W2; G.b2=sum(d2,2);
    d1=(P.W2.'*d2).*(1-A1.^2);
    G.W1=d1*Q.'+weightDecay*P.W1; G.b1=sum(d1,2);
end
end
