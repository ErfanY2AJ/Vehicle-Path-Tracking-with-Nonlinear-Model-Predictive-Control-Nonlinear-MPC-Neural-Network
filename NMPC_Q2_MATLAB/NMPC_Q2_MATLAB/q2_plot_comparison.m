function q2_plot_comparison(rb,rn,opt,folder,label)
%Q2_PLOT_COMPARISON Compare actual states, inputs and timed solver logs.
runs={rb,rn}; colors={'b','r'}; names={'Mathematical NMPC','Neural NMPC'};
f=q2_figure([label ' - actual tracking comparison'],opt);
tiledlayout(2,2,'TileSpacing','compact');
nexttile; plot(rb.ref.Y(1,1:numel(rb.t)),rb.ref.Y(2,1:numel(rb.t)),'k--','LineWidth',1);
hold on;
for j=1:2, r=runs{j}; plot(r.X(1,:),r.X(2,:),colors{j}); end
axis equal; grid on; xlabel('x (m)'); ylabel('y (m)'); legend([{'Reference'},names],'Location','best');
nexttile; plot(rb.t,rb.ref.Y(3,1:numel(rb.t)),'k--'); hold on;
for j=1:2, r=runs{j}; plot(r.t,r.X(3,:),colors{j}); end
grid on; xlabel('Time (s)'); ylabel('Speed (m/s)');
nexttile; hold on;
for j=1:2
    r=runs{j}; e=r.X(1:2,:)-r.ref.Y(1:2,1:numel(r.t));
    plot(r.t,sqrt(sum(e.^2,1)),colors{j});
end
grid on; xlabel('Time (s)'); ylabel('Position error (m)'); legend(names);
nexttile; hold on;
for j=1:2, r=runs{j}; plot(r.t,r.X(3,:)-r.ref.Y(3,1:numel(r.t)),colors{j}); end
grid on; xlabel('Time (s)'); ylabel('Speed error (m/s)');
sgtitle(strrep(label,'_',' '));
q2_save_figure(f,'tracking_comparison',opt,folder);
f=q2_figure([label ' - commands and solver comparison'],opt);
tiledlayout(2,2,'TileSpacing','compact');
nexttile; hold on;
for j=1:2, r=runs{j}; stairs(r.t(1:end-1),r.U(2,:),colors{j}); end
yline(rb.cfg.inputScale(2),'k--'); yline(-rb.cfg.inputScale(2),'k--');
grid on; xlabel('Time (s)'); ylabel('Steering (rad)'); legend(names);
nexttile; hold on;
for j=1:2, r=runs{j}; stairs(r.t(1:end-1),r.U(1,:),colors{j}); end
yline(rb.cfg.inputScale(1),'k--'); yline(-rb.cfg.inputScale(1),'k--');
grid on; xlabel('Time (s)'); ylabel('Commanded torque (N m)');
nexttile; hold on;
for j=1:2, r=runs{j}; plot(r.t(1:end-1),r.log.time,colors{j}); end
yline(rb.cfg.dt,'k--'); grid on; xlabel('Time (s)'); ylabel('Solve time (s)');
nexttile; hold on;
for j=1:2, r=runs{j}; stairs(r.t(1:end-1),r.log.status,colors{j}); end
yticks([1 2 3]); yticklabels({'Converged','Feasible/not converged','Fallback'});
ylim([0.7 3.3]); grid on; xlabel('Time (s)');
q2_save_figure(f,'inputs_solver_comparison',opt,folder);
end
