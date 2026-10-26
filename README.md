# Question 2: Neural Dynamics Inside NMPC — Step-by-Step MATLAB Implementation

This standalone practice project trains a vehicle dynamics model from generated data and replaces the mathematical predictor inside NMPC with that learned model. The actual plant, cost, horizon, constraints, initialization, solver, warm start, and fallback remain identical between the two controllers.

The network, backpropagation, Adam optimizer, dynamics Jacobians, and NMPC cost gradient are implemented explicitly in double precision. **Deep Learning Toolbox is not required.** The default optimizer is the same custom projected BFGS solver used in Question 1.

The original assignment reference-generation source is unavailable. This project therefore retains the explicitly named **practice reference from Question 1**. Its results must be described as practice-reference results.

**Recorded native MATLAB run:** [View the measured results and all eight figures](#recorded-results) from the supplied 15-second quick execution in MATLAB R2025b.

## Step 1 — Unzip and start a short, visible experiment

Set MATLAB's Current Folder to the extracted `NMPC_Q2_MATLAB` folder. Run:

```matlab
q2_self_tests
report = run_nmpc_q2('quick',struct('pauseBetweenSteps',true));
```

The runner pauses **between stages**. Review the figures and Command Window output, then press Enter. Training and closed-loop figures update while calculations are running.

For continuous execution:

```matlab
report = run_nmpc_q2('quick');
```

Requirements: MATLAB R2020a or later for the supplied plotting/export calls. Base MATLAB is sufficient for the default workflow. The optional `fmincon` path requires Optimization Toolbox.

| Stage | Visible or printed result | Saved result |
|---:|---|---|
| 1 | Derivative checks and common-objective equivalence | Configuration |
| 2 | Accepted independent-run counts | Dataset MAT and CSV |
| 3 | Partition table and training histograms | Run IDs and preprocessing |
| 4 | Live training/validation loss and epoch progress | Best validation-selected network |
| 5 | One-step table, recursive-error table, example rollout | Prediction metrics and plots |
| 6 | Live nominal rollouts and paired controller plots | Nominal logs, metrics, and trajectories |
| 7 | Live mass-mismatch rollouts and paired plots | Uncertainty logs, metrics, and trajectories |
| 8 | Combined full/post-transient comparison | Final report and summary CSV |

## Step 2 — Choose quick or standard mode

| Setting | Quick | Standard |
|---|---:|---:|
| Independent data runs | 40 | 60 |
| Consecutive valid transitions per run | 200 | 250 |
| Total transitions | 8000 | 15000 |
| Training/validation/test runs | 28 / 6 / 6 | 42 / 9 / 9 |
| Maximum training epochs | 150 | 400 |
| Hidden neurons | 32 / 32 | 32 / 32 |
| Mini-batch size | 512 | 512 |
| Controller simulation duration per scenario | 15 s | 100 s |
| Prediction horizon | 20 steps | 20 steps |
| Sampling interval | 0.1 s | 0.1 s |

Both modes satisfy the minimum data-run counts. Quick mode is a short workflow preview; use standard mode for the complete 100-second path and later turns:

```matlab
report = run_nmpc_q2('standard',struct('pauseBetweenSteps',true));
```

Training can stop earlier when validation loss fails to improve for the specified patience. Change the budget explicitly if needed:

```matlab
options = struct('maxEpochs',600,'patience',100,'pauseBetweenSteps',true);
report = run_nmpc_q2('standard',options);
```

No execution-time estimate is assumed; runtime depends on the machine, training settings, and number of internal NMPC iterations.

## Step 3 — Generate independent, valid training trajectories

Each stored transition comes directly from the nominal discrete model:

```math
x_{k+1}=f_d(x_k,u_k;p_{\rm nominal}),\qquad
x_k=[p_x,p_y,v,\varphi,T]^{\mathsf T}.
```

**MATLAB — [q2_generate_data.m](q2_generate_data.m)**

```matlab
proposed = vehicle_step(X(:,k),U(:,k),p,dt);
if any(~isfinite(proposed)) || proposed(3)<=0
    valid=false;
    break
end
X(:,k+1)=proposed;
```

The generator:

- Stratifies initial speeds over 5–25 m/s and draws initial headings in `(-pi,pi]`.
- Varies initial position and actual torque between independent runs.
- Uses piecewise random torque and steering, including both normalized endpoints `-1` and `+1`.
- Changes proposed controls near low/high speeds to maintain useful coverage.
- Discards an **entire run** if an invalid transition is proposed, then generates a new run.
- Never clips, repairs, or replaces a resulting state or speed.
- Never uses the mild-path closed-loop test to generate training samples.

Heading stays unwrapped. Wrapping with `atan2(sin(phi),cos(phi))` is used only for the coverage histogram.

The dataset CSV preserves `RunID`, step index, `x`, `u`, and `xNext`. This makes grouping and the exact transition mapping auditable.

## Step 4 — Split complete runs and fit preprocessing on training data only

The split is performed on run IDs, before fitting means, scales, or model coefficients:

```math
\mathcal R_{\rm train}\cap\mathcal R_{\rm val}
=\mathcal R_{\rm train}\cap\mathcal R_{\rm test}
=\mathcal R_{\rm val}\cap\mathcal R_{\rm test}
=\varnothing.
```

**MATLAB — [q2_prepare_data.m](q2_prepare_data.m)**

```matlab
order = randperm(R);
nTrain = floor(opt.trainFraction*R);
nVal = floor(opt.validationFraction*R);
ids = {sort(order(1:nTrain)), ...
       sort(order(nTrain+1:nTrain+nVal)), ...
       sort(order(nTrain+nVal+1:end))};
```

No nearby samples from one physical run can cross partitions. All normalization, affine coefficients, residual scales, constant-output decisions, and feature-range bounds use training runs only. Validation chooses an epoch. Test runs are evaluated afterward.

## Step 5 — Learn body-frame state increments

Absolute position is excluded from the learned model. Instead of the suggested global-increment representation, this implementation uses a body-frame representation, as permitted by the question.

### 5.1 Input features

```math
\eta_k=
\begin{bmatrix}
v_k\\T_k/1965.1\\T_{u,k}/1965.1\\\delta_k/0.5
\end{bmatrix}.
```

**MATLAB — [q2_features.m](q2_features.m)**

```matlab
F = [X(3,:); X(5,:)/scale(1); U(1,:)/scale(1); U(2,:)/scale(2)];
```

Heading is excluded because the learned displacement is expressed in the current body frame. The known coordinate transformation handles heading periodicity; it does not supply the vehicle's learned speed, steering, or actuator mapping.

### 5.2 Training targets

Let `Delta x = xNext-x` and let the rotation matrix be:

```math
R(\varphi)=
\begin{bmatrix}\cos\varphi&-\sin\varphi\\
\sin\varphi&\cos\varphi\end{bmatrix}.
```

The five target increments are:

```math
d_k=
\begin{bmatrix}
R(\varphi_k)^{\mathsf T}(p_{k+1}-p_k)\\
v_{k+1}-v_k\\
\varphi_{k+1}-\varphi_k\\
T_{k+1}-T_k
\end{bmatrix}.
```

**MATLAB — [q2_body_targets.m](q2_body_targets.m)**

```matlab
d = Xnext-X;
c = cos(X(4,:)); s = sin(X(4,:));
D = [c.*d(1,:)+s.*d(2,:); ...
    -s.*d(1,:)+c.*d(2,:);d(3:5,:)];
```

The heading label is the unwrapped difference. It is never wrapped or converted through `atan2`.

## Step 6 — Fit an affine skip and train the residual MLP

### 6.1 Standardize the features

```math
q_k=(\eta_k-\mu_{\rm train})\oslash\sigma_{\rm train}.
```

**MATLAB**

```matlab
mu = mean(train.features,2);
sigma = std(train.features,0,2);
sigma(sigma<1e-12)=1;
q = (train.features-mu)./sigma;
```

### 6.2 Fit the affine skip from data

```math
A_{\rm fit}=\arg\min_A\sum_{k\in{\rm train}}
\left\|d_k-A\begin{bmatrix}1\\q_k\end{bmatrix}\right\|_2^2.
```

**MATLAB**

```matlab
affine = train.targets/[ones(1,size(q,2));q];
residual = train.targets-affine*[ones(1,size(q,2));q];
```

Every coefficient is estimated from the generated data. This is an **affine-skip residual MLP**, rather than a plain MLP without a skip connection. The skip captures linear increment relationships; the network learns the remaining nonlinear relationships.

Residual outputs with training standard deviation below `1e-10` are treated as constants and their nonlinear contribution is set to zero. This threshold avoids dividing floating-point roundoff by an extremely small residual scale. The decision uses training data only.

For this model, body-frame position increments and actuator increments are affine. Their fitted one-step errors can therefore be near machine precision. This does not imply that recursive position error is negligible: heading and speed approximation errors accumulate.

### 6.3 Define the neural mapping

```math
a_1=\tanh(W_1q+b_1),\qquad
a_2=\tanh(W_2a_1+b_2),\qquad
r_\theta=W_3a_2+b_3.
```

**MATLAB — [q2_mlp_forward.m](q2_mlp_forward.m)**

```matlab
A1 = tanh(P.W1*Q+P.b1);
A2 = tanh(P.W2*A1+P.b2);
Y = P.W3*A2+P.b3;
```

The complete increment prediction is:

```math
\widehat d=A_{\rm fit}\begin{bmatrix}1\\q\end{bmatrix}
+\mu_r+\sigma_r\odot r_\theta(q).
```

**MATLAB — [q2_nn_step.m](q2_nn_step.m)**

```matlab
D = net.prep.affine*[ones(1,size(X,2));q]+net.prep.residualMean+ ...
    net.prep.residualScale.*residual;
```

### 6.4 Train with explicit backpropagation and Adam

For active residual heads `I`, the mini-batch objective is:

```math
\mathcal L=
\frac{1}{|I|B}\sum_{i\in I}\sum_{k=1}^{B}
(r_{\theta,i}(q_k)-y_{i,k})^2
+\frac{\lambda}{2}\sum_{\ell=1}^3\|W_\ell\|_F^2,
\qquad \lambda=10^{-6}.
```

**MATLAB — [q2_mlp_loss.m](q2_mlp_loss.m)**

```matlab
[prediction,A1,A2] = q2_mlp_forward(P,Q);
error = (prediction-Y).*active;
denom = max(1,nnz(active))*size(Q,2);
dataLoss = sum(error(:).^2)/denom;
d3 = 2*error/denom;
G.W3 = d3*A2.'+weightDecay*P.W3;
G.b3 = sum(d3,2);
d2 = (P.W3.'*d3).*(1-A2.^2);
G.W2 = d2*A1.'+weightDecay*P.W2;
G.b2 = sum(d2,2);
d1 = (P.W2.'*d2).*(1-A1.^2);
G.W1 = d1*Q.'+weightDecay*P.W1;
G.b1 = sum(d1,2);
```

Adam uses bias-corrected first and second moments:

```math
m_t=0.9m_{t-1}+0.1g_t,\qquad
s_t=0.999s_{t-1}+0.001g_t^2,
```

```math
\widehat m_t=\frac{m_t}{1-0.9^t},\qquad
\widehat s_t=\frac{s_t}{1-0.999^t},\qquad
\theta_{t+1}=\theta_t-\alpha_t
\frac{\widehat m_t}{\sqrt{\widehat s_t}+10^{-8}}.
```

**MATLAB — [q2_adam.m](q2_adam.m)**

```matlab
state.m.(n)=opt.beta1*state.m.(n)+(1-opt.beta1)*gradient;
state.v.(n)=opt.beta2*state.v.(n)+(1-opt.beta2)*gradient.^2;
mhat=state.m.(n)/(1-opt.beta1^step);
vhat=state.v.(n)/(1-opt.beta2^step);
P.(n)=P.(n)-rate*mhat./(sqrt(vhat)+opt.adamEpsilon);
```

The initial learning rate is 0.002, multiplied by 0.3 every 100 epochs. Global gradient norm is clipped to 5. Validation selection uses unregularized residual MSE, not test loss. The returned network is restored to the best validation-selected epoch.

The training graph displays **normalized residual MSE**. It is neither position RMSE in metres nor the closed-loop tracking objective.

## Step 7 — Reconstruct the next global state and its Jacobians

```math
\widehat p_{k+1}=p_k+R(\varphi_k)
\begin{bmatrix}\widehat d_1\\\widehat d_2\end{bmatrix},
\qquad
\begin{bmatrix}\widehat v_{k+1}\\\widehat\varphi_{k+1}\\\widehat T_{k+1}\end{bmatrix}
=\begin{bmatrix}v_k\\\varphi_k\\T_k\end{bmatrix}
+\widehat d_{3:5}.
```

**MATLAB**

```matlab
c = cos(X(4,:)); s = sin(X(4,:));
Xnext = X+[c.*D(1,:)-s.*D(2,:); ...
    s.*D(1,:)+c.*D(2,:);D(3:5,:)];
```

The learned step calls no physical dynamics equation. It uses the fitted parameters, network, and coordinate reconstruction.

For the normalized-feature network, its increment Jacobian is:

```math
\frac{\partial\widehat d}{\partial q}
=A_{{\rm fit},:,2:5}
+D_{\sigma_r}W_3D_{1-a_2^2}W_2D_{1-a_1^2}W_1.
```

Apply feature scaling, body-to-global rotation, and the identity from the residual state update to obtain `A = d(xNext)/dx` and `B = d(xNext)/du`.

**MATLAB — analytical chain rule**

```matlab
tanhDerivative2 = 1-A2.^2;
tanhDerivative1 = 1-A1.^2;
Jhidden = P.W3.*tanhDerivative2.';
Jhidden = (Jhidden*P.W2).*tanhDerivative1.';
J = net.prep.affine(:,2:end)+net.prep.residualScale.*(Jhidden*P.W1);
J = J./net.prep.featureStd.';
```

The remaining reconstruction derivatives are implemented in [q2_nn_step.m](q2_nn_step.m). [q2_self_tests.m](q2_self_tests.m) compares both Jacobians and the full NMPC gradient against central differences.

## Step 8 — Evaluate held-out predictions

### 8.1 One-step metrics

For every held-out transition, compare `xNext` with the network's prediction from the recorded current state and command. The output table reports RMSE and MAE separately for:

| State | Unit |
|---|---|
| `px`, `py` | m |
| `v` | m/s |
| `phi` | rad, unwrapped |
| `T` | N·m |

The persistence baseline predicts `xNext = x`. The reported R² is calculated on **state increments**, not absolute next states.

**MATLAB — inspect the table**

```matlab
report.evaluation.oneStep
```

### 8.2 Recursive metrics

For each `H = 1,5,10,20` and each test run with `K` commands, use all starting points `s = 1:K-H+1`. Supply the actual initial state and recorded future commands, then repeatedly use the network's **own predicted state**:

```math
\widehat x_s=x_s,\qquad
\widehat x_{s+j+1}=f_\theta(\widehat x_{s+j},u_{s+j}).
```

**MATLAB — [q2_evaluate_network.m](q2_evaluate_network.m)**

```matlab
starts = 1:K-H+1;
X = run.X(:,starts);
for j=1:H
    X = q2_nn_step(X,run.U(:,starts+j-1),net);
end
truth = run.X(:,starts+H);
```

Each horizon reports mean Euclidean endpoint position error and mean absolute endpoint speed error, along with the persistence baseline and number of starting points. Nonfinite endpoints are counted and cause the corresponding neural aggregate errors to be infinite rather than silently excluded.

**MATLAB**

```matlab
report.evaluation.recursive
```

These are prediction errors on random-control test runs. They are distinct from tracking errors on the closed-loop practice path.

## Step 9 — Compare controllers with only the predictor changed

The two controllers call one common cost implementation, [q2_objective.m](q2_objective.m):

```math
J=\sum_{j=1}^{N_p}e_j^{\mathsf T}Qe_j
+e_{N_p}^{\mathsf T}Pe_{N_p}
+\sum_{j=0}^{N_p-1}(z_j-z_j^{\rm ref})^{\mathsf T}R(z_j-z_j^{\rm ref})
+\sum_{j=0}^{N_p-1}\Delta z_j^{\mathsf T}S\Delta z_j.
```

The terminal state receives both `Q` and `P`, exactly as in Question 1. Both use normalized command bounds `[-1,1]` and the same analytical adjoint gradient.

**MATLAB — predictor dispatch in [q2_predictor_step.m](q2_predictor_step.m)**

```matlab
% Mathematical controller:
[xn,A,B] = vehicle_step(x,u,cfg.p,cfg.dt);
% Neural controller:
[xn,A,B] = q2_nn_step(x,u,cfg.net);
```

The measured actual state initializes every horizon. Only the first selected command is applied to `vehicle_step` with the actual plant parameters.

Both controllers use:

- The mild practice path, the same initial state, and direct tracking.
- `Np=20`, `dt=0.1`, and the original Question 1 weight matrices by default.
- `maxIter=200`, `gradTol=1e-6`, `stepTol=1e-10`, and identical line-search settings.
- The same sequence shift for warm starts.
- The same candidate validity, cost comparison, and fallback rules.

To use an exact saved Question 1 configuration, supply that complete `bestCfg` to both methods:

```matlab
options = struct('controller',bestCfg,'pauseBetweenSteps',true);
report = run_nmpc_q2('standard',options);
```

This starts a fresh dataset and training run at that configuration's sampling time. A saved network cannot be reused with a different `dt`.

## Step 10 — Read live plots and the comparison table

For each case, the runner shows actual paths and speeds, position and speed tracking errors, commanded steering and torque, solution time, and solver status.

**MATLAB**

```matlab
report.nominal.summary
report.uncertainty.summary
report.summary
```

Every controller has a `full` row and an `after5s` row. Each contains position/speed RMSE and MAE, normalized effort `Eu`, input variation `EdeltaU`, mean/max solution time, and all three counters.

| Status | Interpretation |
|---:|---|
| 1 | Valid candidate; solver reported convergence |
| 2 | Feasible candidate; gradient convergence not reached; cost no worse than warm start |
| 3 | Warm-start fallback |

Actual-state feature extrapolation is monitored against the training ranges for `v,T,Tu,delta`. This is an axis-aligned range diagnostic; it does not certify that an actual feature combination is well represented by training data. It does not impose a new constraint or silently switch predictors.

Timing includes warm-cost evaluation, optimization, and candidate selection. It excludes plant updates, extrapolation diagnostics, live plotting, and file writes. Both methods use analytical gradients, although evaluation costs differ. Nominal timing runs mathematical first; the uncertainty case reverses that order. This is a single observed pairing, not a repeated timing benchmark.

## Step 11 — Apply the same plant uncertainty to both controllers

The default uncertainty experiment sets:

```math
m_{\rm actual}=1.1(1270)=1397\ {\rm kg},
\qquad
m_{\rm mathematical\ predictor}=1270\ {\rm kg}.
```

The neural model is the same network trained on nominal data. It is not retrained on mass-mismatch trajectories.

**MATLAB — actual plant parameter**

```matlab
actual = cfg.p;
actual.m = cfg.p.m*cfg.actualMassFactor;
X(:,k+1) = vehicle_step(X(:,k),U(:,k),actual,cfg.dt);
```

Compare nominal and uncertain rows separately for each method. Neural approximation error and plant-parameter mismatch are different error sources.

The neural model approximates the nominal mathematical model. Better nominal tracking is not presumed. Any apparent benefit must be interpreted using prediction errors, solver behavior, and measured timing rather than claimed as inherent neural-controller superiority.

## Step 12 — Execute stages manually or reuse a saved network

For direct control over execution, run these blocks in order from the project root.

**1. Configuration and tests**

```matlab
q2_self_tests
opt = q2_config('quick');
if ~exist(opt.outputRoot,'dir'), mkdir(opt.outputRoot); end
folder = tempname(opt.outputRoot);
mkdir(folder);
```

**2. Data generation**

```matlab
data = q2_generate_data(opt,folder);
```

**3. Run splitting, preprocessing, and coverage**

```matlab
split = q2_prepare_data(data,opt,folder);
```

**4. Network training**

```matlab
[net,history] = q2_train_network(split,opt,folder);
```

**5. Prediction evaluation**

```matlab
evaluation = q2_evaluate_network(data,split,net,opt,folder);
```

**6. Nominal controller comparison**

```matlab
nominal = q2_compare_pair(opt,net,folder,'nominal_mild',1,false);
```

**7. Mass-mismatch comparison**

```matlab
uncertain = q2_compare_pair(opt,net,folder,'mass_plus10pct',1.1,true);
```

**8. Combined table**

```matlab
summary = [nominal.summary;uncertain.summary];
disp(summary);
```

To reuse a saved model, select its actual MAT file:

```matlab
[file,path] = uigetfile('step04_trained_network.mat','Select trained Question 2 network');
if isequal(file,0), return; end
saved = load(fullfile(path,file),'net','opt');
net = saved.net;
opt = saved.opt;
opt.simDuration = 100;
pair = q2_compare_pair(opt,net,path,'nominal_recheck',1,false);
```

Reusing this network at the same `dt` avoids repeating training. For retraining, reload `step03_partitions.mat` and run `q2_train_network`. Training-epoch resumption is not implemented.

## Step 13 — Find the saved results

All new outputs go into a fresh timestamped directory under `results_q2`.

| Output | Contents |
|---|---|
| `step01_options.mat` | Complete experiment configuration |
| `step02_dataset.mat`, `dataset.csv` | Independent nominal runs and exact transitions |
| `step03_partitions.mat` | Grouped splits and training-only preprocessing |
| `run_partitions.csv` | Run ID and partition code: 1 train, 2 validation, 3 test |
| `partition_coverage.csv` | Coverage and sample counts by partition |
| `step03_training_coverage.png` | Histograms and speed/command scatter |
| `step04_trained_network.mat` | Best network, loss history, and options |
| `training_history.csv`, `step04_training_loss.png` | Epoch-by-epoch losses |
| `one_step_test_metrics.csv` | Per-state next-step metrics |
| `recursive_test_metrics.csv` | Recursive endpoint metrics for all horizons |
| `step05_prediction_evaluation.mat` | Prediction-evaluation tables and test IDs |
| `step05_prediction_errors.png` | Error curves and persistence comparison |
| `step05_recursive_example.png` | A held-out recursive rollout example |
| `nominal_mild`, `mass_plus10pct` | Paired controller trajectories, plots, CSV logs, and summary |
| `final_comparison.csv` | Full/post-transient controller comparison |
| `q2_report.mat` | Final report plus trained network |
| `experiment_notes.txt` | MATLAB version, reference, and timing definitions |

Each controller subdirectory contains `result.mat`, `states.csv`, `commands.csv`, `solver_log.csv`, and `run_notes.txt`. The solver log retains exit flag, stop message, iterations, evaluation count, optimality, costs, time, and command-selection status.

## Step 14 — Verification and interpretation limits

The authoring environment did not execute MATLAB. The supplied archive now provides native MATLAB R2025b outputs for the quick workflow, documented in Step 15. These are distinct from the earlier syntax checks and independent NumPy verification. The independent implementation checked:

- MLP backpropagation against finite differences.
- Both learned state/input Jacobians against finite differences.
- The Jacobian expressions extracted from the delivered MATLAB file, including a check that rejects a missing square in either tanh derivative.
- The full neural single-shooting objective gradient.
- Translation equivariance and heading periodicity.
- Generation of 40 valid, independent 200-step runs and a 28/6/6 split.
- Learning and recursive evaluation in an independent numerical execution.
- Separate 15-second closed-loop checks for both predictors, nominal and mass +10%, with finite positive-speed states and bounded commands.

The script and its numerical verification output are in [validation/independent_q2_check.py](validation/independent_q2_check.py) and [validation/independent_verification.json](validation/independent_verification.json). The optional [closed-loop check](validation/independent_closed_loop_check.py) and its [recorded output](validation/independent_closed_loop_results.json) are also included. These are **independent NumPy checks, not MATLAB results**. The bundled network is not pretrained: the MATLAB workflow generates data and trains its own network.

The additional [MATLAB-expression regression check](validation/check_matlab_jacobian_source.py) checks 40 random networks; its [recorded output](validation/matlab_expression_check.json) identifies the omitted-square regression. It evaluates a limited numeric expression subset and is not a MATLAB runtime.

Run `q2_self_tests` locally before an experiment. The supplied run records graphics, CSV/MAT export, training, prediction evaluation, and both default controller comparisons. It does not test the optional `fmincon` path or the 100-second standard workflow. Step 15 transcribes the provided native MATLAB outputs.

The practice plant is a kinematic vehicle model without tire-slip dynamics or measurement noise. Neither low prediction error, command feasibility, nor a converged local solver certifies global optimality, physical safety, closed-loop stability, or hard real-time performance.

Implementation references:

- [MathWorks: tanh](https://www.mathworks.com/help/matlab/ref/double.tanh.html)
- [MathWorks: drawnow](https://www.mathworks.com/help/matlab/ref/drawnow.html)
- [MathWorks: Adam update equations](https://www.mathworks.com/help/deeplearning/ref/adamupdate.html) — background only; this project implements the updates directly and does not call `adamupdate`.

<a id="recorded-results"></a>
## Step 15 — Recorded MATLAB results

### 15.1 Execution and image paths

These are the supplied native MATLAB results, distinct from the earlier independent NumPy checks.

| Setting | Recorded value |
|---|---|
| MATLAB | 25.2.0.2998904 (R2025b) |
| Mode / seed | quick / 42 |
| Data | 40 independent runs, 200 valid transitions per run |
| Train / validation / test | 28 / 6 / 6 runs; 5600 / 1200 / 1200 transitions |
| Network | 4–32–32–5; tanh hidden layers; affine skip |
| Sampling / prediction horizon | 0.1 s / 20 steps = 2 s |
| Closed-loop duration | 15 s per controller and scenario |
| Reference | Mild practice path, alpha = 0.3 |
| Actual plant mass | 1270 kg nominal; 1397 kg with uncertainty |
| Solver | projected-bfgs; gradTol = 1e-6; stepTol = 1e-10 |

All four runs completed, each with 151 state samples and 150 commands. The after-5-second window has 101 states and 100 commands.

**Repository layout:** keep this README inside `NMPC_Q2_MATLAB`, with the extracted `results_q2` directory alongside it. Images below use relative paths into `results_q2/q2_quick_20261026_233935`. GitHub cannot display images stored only inside a ZIP; extract and upload the files. If using Git, remove the `results_q2/` ignore rule or explicitly add this recorded directory.

This 15-second preview covers the initial transient and first turn. It does not cover all later segments of the 100-second path. Full-run averages from different simulation durations should not be directly compared.


Sources: [experiment_notes.txt](results_q2/q2_quick_20261026_233935/experiment_notes.txt) and [step01_options.mat](results_q2/q2_quick_20261026_233935/step01_options.mat).

### 15.2 Data coverage and training

Whole runs were split before preprocessing. Each partition covers [-1,1] in both normalized command components. Observed speeds remain positive; state speeds can extend beyond the initial-speed sampling interval of 5–25 m/s.

| Partition | Runs | Transitions | Minimum speed (m/s) | Maximum speed (m/s) |
| --- | --- | --- | --- | --- |
| train | 28 | 5600 | 4.81006 | 26.074 |
| validation | 6 | 1200 | 4.84338 | 25.7637 |
| test | 6 | 1200 | 5.09574 | 25.9071 |

![Training-only input and state coverage](results_q2/q2_quick_20261026_233935/step03_training_coverage.png)

Source: [partition_coverage.csv](results_q2/q2_quick_20261026_233935/partition_coverage.csv).

The selected epoch was **150**, the final epoch. Training residual MSE decreased from 1.00148 to 0.00034124; validation residual MSE decreased from 0.937948 to **0.000294378**. The recorded training-loop time was 4.10343 s, excluding data generation and controller simulation. These losses use normalized residual targets, not tracking error in metres. Reaching the epoch limit does not establish that further training cannot improve the model.

![Native MATLAB training and validation loss](results_q2/q2_quick_20261026_233935/step04_training_loss.png)

Sources: [training_history.csv](results_q2/q2_quick_20261026_233935/training_history.csv); [step04_trained_network.mat](results_q2/q2_quick_20261026_233935/step04_trained_network.mat).

### 15.3 One-step predictions on held-out runs

RMSE and MAE compare predicted and actual next states. R² is calculated on state increments.

| State | Unit | Neural RMSE | Neural MAE | Persistence RMSE | Persistence MAE | Increment R² |
| --- | --- | --- | --- | --- | --- | --- |
| px | m | 1.99283e-15 | 2.0627e-16 | 1.22294 | 1.04551 | 1 |
| py | m | 2.0808e-15 | 1.88553e-16 | 1.24855 | 1.06098 | 1 |
| v | m/s | 3.83113e-05 | 2.84127e-05 | 0.259424 | 0.226251 | 1 |
| phi | rad | 0.002698 | 0.00179477 | 0.352172 | 0.296133 | 0.999941 |
| T | N m | 1.46365e-13 | 9.59233e-14 | 226.07 | 167.88 | 1 |

Source: [one_step_test_metrics.csv](results_q2/q2_quick_20261026_233935/one_step_test_metrics.csv). Body-frame position and actuator increments are affine in this representation, explaining their near-machine-precision one-step fit. Heading and speed retain approximation errors, which can accumulate recursively.

### 15.4 Recursive prediction

All starts with sufficient remaining commands were evaluated independently at each horizon. Predicted intermediate states were never replaced with measured states.

| H (steps) | Duration (s) | Starts | Neural position error (m) | Persistence (m) | Neural speed error (m/s) | Persistence (m/s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1 | 1200 | 3.93739e-16 | 1.65514 | 2.84127e-05 | 0.226251 |
| 5 | 0.5 | 1176 | 0.0293259 | 7.26504 | 0.000124181 | 1.09303 |
| 10 | 1 | 1146 | 0.0978272 | 11.063 | 0.000205859 | 1.96534 |
| 20 | 2 | 1086 | 0.249434 | 15.5698 | 0.000307081 | 3.23373 |

Position errors are mean Euclidean endpoint errors; speed errors are mean absolute endpoint errors. All horizons reported zero nonfinite and zero nonpositive-speed endpoints. At 20 steps (2 s), neural mean position error reached **0.249434 m** and mean absolute speed error was **0.000307081 m/s**. Heading error accumulates through successive position updates, despite nearly exact one-step position reconstruction.

![Held-out prediction errors and recursive comparison with persistence](results_q2/q2_quick_20261026_233935/step05_prediction_errors.png)

![Recorded test trajectory versus recursive neural prediction](results_q2/q2_quick_20261026_233935/step05_recursive_example.png)

Source: [recursive_test_metrics.csv](results_q2/q2_quick_20261026_233935/recursive_test_metrics.csv).

### 15.5 Closed-loop tracking

Position metrics use radial two-dimensional error. The full interval and post-transient interval are reported separately.

**Full interval: 0–15 s**

| Case | Predictor | Position RMSE (m) | Position MAE (m) | Speed RMSE (m/s) | Speed MAE (m/s) |
| --- | --- | --- | --- | --- | --- |
| Nominal mild | Mathematical | 0.155966 | 0.0694585 | 0.110944 | 0.0479705 |
| Nominal mild | Neural | 0.156009 | 0.0705278 | 0.110927 | 0.0479905 |
| Mass +10% | Mathematical | 0.157247 | 0.0681341 | 0.114963 | 0.0492815 |
| Mass +10% | Neural | 0.157283 | 0.0685837 | 0.114946 | 0.0493016 |

**Post-transient interval: 5–15 s**

| Case | Predictor | Position RMSE (m) | Position MAE (m) | Speed RMSE (m/s) | Speed MAE (m/s) |
| --- | --- | --- | --- | --- | --- |
| Nominal mild | Mathematical | 0.00506277 | 0.00269333 | 0.00368637 | 0.00188282 |
| Nominal mild | Neural | 0.00570137 | 0.00422664 | 0.00368612 | 0.00192736 |
| Mass +10% | Mathematical | 0.00509302 | 0.00484594 | 0.0028227 | 0.00142194 |
| Mass +10% | Neural | 0.0055512 | 0.00543932 | 0.00282285 | 0.00146669 |

The neural post-transient position RMSE was **12.6% higher** under nominal conditions and **9.0% higher** with mass +10%. Speed errors were very close. This run shows comparable tracking with a modest position-accuracy penalty for the learned predictor.

The 20-step random-control test error and closed-loop tracking RMSE describe different experiments: the controller follows a mild path and resets each optimization to the newly observed actual state. Their magnitudes should not be equated.

![Nominal tracking: reference and both actual controller trajectories](results_q2/q2_quick_20261026_233935/nominal_mild/tracking_comparison.png)

![Mass +10 percent: reference and both actual controller trajectories](results_q2/q2_quick_20261026_233935/mass_plus10pct/tracking_comparison.png)

### 15.6 Control effort and smoothness

Both quantities are unweighted means in normalized command coordinates; they are not physical energy or time derivatives.

| Case | Predictor | Window | Eu | EdeltaU |
| --- | --- | --- | --- | --- |
| Nominal mild | Mathematical | full | 0.0278448 | 0.0115277 |
| Nominal mild | Mathematical | after5s | 0.0001299 | 8.01692e-07 |
| Nominal mild | Neural | full | 0.0278406 | 0.0115274 |
| Nominal mild | Neural | after5s | 0.000130013 | 8.12917e-07 |
| Mass +10% | Mathematical | full | 0.0285166 | 0.0111609 |
| Mass +10% | Mathematical | after5s | 0.000134394 | 7.93332e-07 |
| Mass +10% | Neural | full | 0.0285124 | 0.0111607 |
| Mass +10% | Neural | after5s | 0.000134496 | 8.04263e-07 |

### 15.7 Solve time, convergence, and feasibility

The following values cover all 150 command steps. Steps exceeding 100 ms were counted directly from the saved solver logs.

| Case | Predictor | Mean (ms) | Maximum (ms) | Converged | Feasible/not converged | Fallback | Bound violations | Steps >100 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nominal mild | Mathematical | 14.107 | 101.626 | 150 | 0 | 0 | 0 | 1 |
| Nominal mild | Neural | 78.582 | 525.554 | 138 | 12 | 0 | 0 | 5 |
| Mass +10% | Mathematical | 15.814 | 124.916 | 148 | 2 | 0 | 0 | 1 |
| Mass +10% | Neural | 71.324 | 657.902 | 135 | 15 | 0 | 0 | 3 |

The neural mean solve time was **5.57 times** the mathematical mean in the nominal case and **4.51 times** in the mass-mismatch case. Mean times were below the 100 ms sampling interval, but individual steps exceeded it. These observations do not establish hard real-time feasibility.

All runs reported zero fallback commands, zero bound violations, and zero nonfinite applied commands. The actual-state feature-range diagnostic was zero in every run; marginal range coverage does not certify sufficient joint training coverage.

Neural nonconvergence messages included `Maximum iterations reached` and `Small step; gradient tolerance not satisfied`. Their feasible candidates were retained according to the unchanged cost-based selection rule.

![Nominal commands, solution times, and solver status](results_q2/q2_quick_20261026_233935/nominal_mild/inputs_solver_comparison.png)

![Mass +10 percent commands, solution times, and solver status](results_q2/q2_quick_20261026_233935/mass_plus10pct/inputs_solver_comparison.png)

### 15.8 Recorded files

- [Complete controller comparison CSV](results_q2/q2_quick_20261026_233935/final_comparison.csv)
- [Nominal paired MATLAB results](results_q2/q2_quick_20261026_233935/nominal_mild/paired_results.mat)
- [Mass-mismatch paired MATLAB results](results_q2/q2_quick_20261026_233935/mass_plus10pct/paired_results.mat)
- [Complete MATLAB report and trained network](results_q2/q2_quick_20261026_233935/q2_report.mat)
- [Run-to-partition assignment](results_q2/q2_quick_20261026_233935/run_partitions.csv)

Values in this section are rounded from the supplied CSV/MAT files. The raw files retain their recorded precision. No simulations were rerun to construct this results section.
