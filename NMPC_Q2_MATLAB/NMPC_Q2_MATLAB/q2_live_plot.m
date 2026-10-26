function live = q2_live_plot(action,live,X,U,ref,logData,k,cfg,opt)
%Q2_LIVE_PLOT Graphics are updated OUTSIDE the timed optimization section.
if strcmp(action,'create')
    live=struct;
    if ~opt.showFigures, return; end
    live.f=q2_figure(['Live closed loop - ' cfg.predictorType],opt);
    tiledlayout(2,2,'TileSpacing','compact');
    nexttile; plot(ref.Y(1,1:ref.K+1),ref.Y(2,1:ref.K+1),'k--'); hold on;
    live.path=plot(X(1,1),X(2,1),'b-'); axis equal; grid on;
    xlabel('x (m)'); ylabel('y (m)'); legend('Reference','Actual');
    nexttile; live.err=plot(0,norm(X(1:2,1)-ref.Y(1:2,1))); grid on;
    xlabel('Time (s)'); ylabel('Position error (m)');
    nexttile; live.steer=stairs(0,0); grid on;
    xlabel('Time (s)'); ylabel('Steering (rad)');
    nexttile; live.time=plot(0,0); hold on; yline(cfg.dt,'r--');
    grid on; xlabel('Command time (s)'); ylabel('Solve time (s)');
    drawnow;
else
    if ~isfield(live,'f') || ~isgraphics(live.f), return; end
    t=(0:k)*cfg.dt; tu=(0:k-1)*cfg.dt;
    error=sqrt(sum((X(1:2,1:k+1)-ref.Y(1:2,1:k+1)).^2,1));
    set(live.path,'XData',X(1,1:k+1),'YData',X(2,1:k+1));
    set(live.err,'XData',t,'YData',error);
    set(live.steer,'XData',tu,'YData',U(2,1:k));
    set(live.time,'XData',tu,'YData',logData.time(1:k));
    drawnow limitrate;
end
end
