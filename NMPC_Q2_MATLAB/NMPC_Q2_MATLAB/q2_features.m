function F = q2_features(X,U,scale)
%Q2_FEATURES Rotation/translation-invariant body-frame model features.
% X: 5-by-M, U: 2-by-M. Absolute position/heading are not network inputs.
F=[X(3,:);X(5,:)/scale(1);U(1,:)/scale(1);U(2,:)/scale(2)];
end
