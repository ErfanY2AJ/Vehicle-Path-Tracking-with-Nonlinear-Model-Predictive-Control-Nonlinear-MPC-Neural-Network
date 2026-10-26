function D = q2_body_targets(X,Xnext)
%Q2_BODY_TARGETS Transform exact GLOBAL increments into the current body frame.
% Angular labels use the unwrapped difference, never atan2/wrapToPi.
d=Xnext-X; c=cos(X(4,:)); s=sin(X(4,:));
D=[c.*d(1,:)+s.*d(2,:);-s.*d(1,:)+c.*d(2,:);d(3:5,:)];
end
