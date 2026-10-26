function [Y,A1,A2] = q2_mlp_forward(P,Q)
%Q2_MLP_FORWARD Columns are samples. Base MATLAB, double precision.
A1=tanh(P.W1*Q+P.b1);
A2=tanh(P.W2*A1+P.b2);
Y=P.W3*A2+P.b3;
end
