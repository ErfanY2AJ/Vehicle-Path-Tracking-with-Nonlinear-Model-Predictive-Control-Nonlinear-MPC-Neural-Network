function f = q2_figure(name,opt)
%Q2_FIGURE One independently named window per experiment stage.
visible='off'; if opt.showFigures, visible='on'; end
f=figure('Name',name,'NumberTitle','off','Color','w','Visible',visible);
end
