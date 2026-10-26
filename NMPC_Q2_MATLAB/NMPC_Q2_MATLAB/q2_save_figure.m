function q2_save_figure(f,name,opt,folder)
%Q2_SAVE_FIGURE Persist figures without requiring a graphics add-on.
drawnow;
if opt.saveFigures
    exportgraphics(f,fullfile(folder,[name '.png']),'Resolution',160);
end
if ~opt.showFigures, close(f); end
end
