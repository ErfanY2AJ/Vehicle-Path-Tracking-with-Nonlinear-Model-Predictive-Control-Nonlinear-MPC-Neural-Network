function [net,history] = q2_train_network(split,opt,folder)
%Q2_TRAIN_NETWORK Fit ONLY training runs; select epoch ONLY on validation.
old=rng; cleanup=onCleanup(@()rng(old)); %#ok<NASGU>
rng(opt.seed+2);
active=split.prep.active;
assert(any(active),'All training dynamics were affine; no residual network is needed.');
P=q2_mlp_init(opt.hidden,active); best=P; state=[]; step=0; stale=0;
qt=split.train.Q; yt=split.train.Y;
qv=split.validation.Q; yv=split.validation.Y;
initialTrain=q2_mlp_loss(P,qt,yt,active,0);
initialVal=q2_mlp_loss(P,qv,yv,active,0);
bestVal=initialVal; bestEpoch=0;
record=nan(opt.maxEpochs+1,4);
record(1,:)=[0,initialTrain,initialVal,0];
f=q2_figure('Step 4 - live training and validation loss',opt);
ax=axes(f); trainLine=semilogy(ax,0,max(initialTrain,realmin),'b-','LineWidth',1.2);
hold(ax,'on'); valLine=semilogy(ax,0,max(initialVal,realmin),'r-','LineWidth',1.2);
grid(ax,'on'); xlabel(ax,'Epoch'); ylabel(ax,'Normalized residual MSE');
legend(ax,'Training runs','Validation runs','Location','best');
title(ax,'Tanh residual MLP with a training-fitted affine skip');
drawnow; clock=tic; lastEpoch=0;
fprintf('  Network: 4-%d-%d-5, tanh/tanh/linear, affine skip.\n',opt.hidden);
fprintf('  Active nonlinear output heads: %s\n',mat2str(find(active).'));
for epoch=1:opt.maxEpochs
    order=randperm(size(qt,2));
    rate=opt.learningRate*0.3^floor((epoch-1)/100);
    for first=1:opt.batchSize:numel(order)
        idx=order(first:min(first+opt.batchSize-1,numel(order)));
        [loss,G]=q2_mlp_loss(P,qt(:,idx),yt(:,idx),active,opt.weightDecay);
        if ~isfinite(loss), error('Training loss became nonfinite.'); end
        step=step+1; [P,state]=q2_adam(P,G,state,step,rate,opt);
    end
    trainLoss=q2_mlp_loss(P,qt,yt,active,0);
    valLoss=q2_mlp_loss(P,qv,yv,active,0);
    record(epoch+1,:)=[epoch,trainLoss,valLoss,toc(clock)];
    if valLoss<bestVal-opt.minImprovement
        best=P; bestVal=valLoss; bestEpoch=epoch; stale=0;
    else
        stale=stale+1;
    end
    lastEpoch=epoch;
    if isgraphics(f)
        set(trainLine,'XData',record(1:epoch+1,1),'YData',max(record(1:epoch+1,2),realmin));
        set(valLine,'XData',record(1:epoch+1,1),'YData',max(record(1:epoch+1,3),realmin));
        title(ax,sprintf('Epoch %d | validation %.3g | best epoch %d',epoch,valLoss,bestEpoch));
        drawnow limitrate;
    end
    if epoch==1 || mod(epoch,opt.printEvery)==0
        fprintf('  Epoch %4d/%d: train=%.5g; validation=%.5g; best=%d\n', ...
            epoch,opt.maxEpochs,trainLoss,valLoss,bestEpoch);
    end
    if stale>=opt.patience
        fprintf('  Early stopping: no validation improvement for %d epochs.\n',stale);
        break
    end
end
net=struct('params',best,'prep',split.prep,'inputScale',split.inputScale, ...
    'dt',split.dt,'trainingParameters',split.p,'hidden',opt.hidden, ...
    'bestEpoch',bestEpoch,'bestValidationLoss',bestVal, ...
    'trainingRunIDs',split.runIDs{1},'validationRunIDs',split.runIDs{2}, ...
    'representation','Body-frame increments; affine skip plus residual tanh MLP');
history=array2table(record(1:lastEpoch+1,:), ...
    'VariableNames',{'Epoch','TrainMSE','ValidationMSE','Elapsed_s'});
writetable(history,fullfile(folder,'training_history.csv'));
save(fullfile(folder,'step04_trained_network.mat'),'net','history','opt','-v7');
if isgraphics(f), q2_save_figure(f,'step04_training_loss',opt,folder); end
fprintf('  Restored validation-selected epoch %d; test runs have not been used for fitting.\n',bestEpoch);
end
