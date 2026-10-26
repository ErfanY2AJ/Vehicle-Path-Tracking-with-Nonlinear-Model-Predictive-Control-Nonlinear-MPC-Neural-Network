function q2_stage(number,titleText,opt)
%Q2_STAGE Print stage boundaries; optional pauses happen BETWEEN stages.
if number>1 && opt.pauseBetweenSteps
    input('Review the current results, then press Enter to continue. ','s');
end
fprintf('\nSTEP %d: %s\n',number,titleText);
drawnow;
end
