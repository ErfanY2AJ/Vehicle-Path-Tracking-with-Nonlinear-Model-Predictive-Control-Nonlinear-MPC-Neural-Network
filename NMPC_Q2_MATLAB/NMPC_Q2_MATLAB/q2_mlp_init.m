function P = q2_mlp_init(hidden,active)
%Q2_MLP_INIT Xavier initialization; two tanh hidden layers, linear head.
h1=hidden(1); h2=hidden(2);
P.W1=sqrt(6/(4+h1))*(2*rand(h1,4)-1); P.b1=zeros(h1,1);
P.W2=sqrt(6/(h1+h2))*(2*rand(h2,h1)-1); P.b2=zeros(h2,1);
P.W3=0.1*sqrt(6/(h2+5))*(2*rand(5,h2)-1); P.b3=zeros(5,1);
P.W3(~active,:)=0;
end
