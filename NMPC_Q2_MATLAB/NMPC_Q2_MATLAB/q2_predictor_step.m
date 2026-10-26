function [xn,A,B] = q2_predictor_step(x,u,cfg)
%Q2_PREDICTOR_STEP Only this internal predictor differs between controllers.
if strcmp(cfg.predictorType,'math')
    if nargout>1, [xn,A,B]=vehicle_step(x,u,cfg.p,cfg.dt);
    else, xn=vehicle_step(x,u,cfg.p,cfg.dt); end
elseif strcmp(cfg.predictorType,'neural')
    assert(abs(cfg.dt-cfg.net.dt)<1e-12, ...
        'Sampling time changed: retrain the neural model at this dt.');
    if nargout>1, [xn,A,B]=q2_nn_step(x,u,cfg.net);
    else, xn=q2_nn_step(x,u,cfg.net); end
else
    error('Unknown predictor type.');
end
end
