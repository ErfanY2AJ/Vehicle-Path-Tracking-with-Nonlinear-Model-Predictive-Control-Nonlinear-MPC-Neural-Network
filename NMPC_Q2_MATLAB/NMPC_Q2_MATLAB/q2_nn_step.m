function [Xnext,A,B] = q2_nn_step(X,U,net)
%Q2_NN_STEP Learned dynamics ONLY: feature map, fitted model, reconstruction.
% Batch columns are supported when only Xnext is requested.
% A/B are single-sample analytical Jacobians wrt x and PHYSICAL u.
if nargout>1 && size(X,2)~=1, error('Jacobians require one sample.'); end
F=q2_features(X,U,net.inputScale);
q=(F-net.prep.featureMean)./net.prep.featureStd;
[residual,A1,A2]=q2_mlp_forward(net.params,q);
D=net.prep.affine*[ones(1,size(X,2));q]+net.prep.residualMean+ ...
    net.prep.residualScale.*residual;
c=cos(X(4,:)); s=sin(X(4,:));
Xnext=X+[c.*D(1,:)-s.*D(2,:);s.*D(1,:)+c.*D(2,:);D(3:5,:)];
if nargout>1
    P=net.params;
    % Column/row scaling avoids allocating hidden-layer diagonal matrices.
    tanhDerivative2=1-A2.^2;
    tanhDerivative1=1-A1.^2;
    Jhidden=P.W3.*tanhDerivative2.';
    Jhidden=(Jhidden*P.W2).*tanhDerivative1.';
    J=net.prep.affine(:,2:end)+net.prep.residualScale.*(Jhidden*P.W1);
    J=J./net.prep.featureStd.';
    rotate=eye(5); rotate(1:2,1:2)=[c,-s;s,c];
    Jglobal=rotate*J;
    A=eye(5);
    A(:,3)=A(:,3)+Jglobal(:,1);
    A(:,5)=A(:,5)+Jglobal(:,2)/net.inputScale(1);
    A(1,4)=-s*D(1)-c*D(2);
    A(2,4)=c*D(1)-s*D(2);
    B=[Jglobal(:,3)/net.inputScale(1),Jglobal(:,4)/net.inputScale(2)];
end
end
