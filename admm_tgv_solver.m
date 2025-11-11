% =========================================================================
%               admm_tgv_solver.m
% =========================================================================
% 作用: 使用 ADMM 算法求解包含 TGV 正则化的 MAP 重建问题。
%      min_{x,w} f(x) + alpha1 * ||Dx - w||_1 + alpha0 * ||Ew||_1
%      其中 f(x) 是数据保真项, D 是梯度算子, E 是对称梯度算子。
% =========================================================================
function [ res ] = admm_tgv_solver(h, mask, x0, scz, sizeI, ...
                                  lambda_residual, alpha1_weights, alpha0_weights, warmstart, ...
                                  N, xt, gm, QE, I, amb, DC, gpu, ...
                                  max_it, verbose)

    % --- 1. 初始化 ---
    
    % 权重参数
    % alpha1_weights: [alpha1_depth, alpha1_albedo]
    % alpha0_weights: [alpha0_depth, alpha0_albedo]
    
    % 变量归一化
    x0(:,:,1) = x0(:,:,1)/scz;

    % 数据整形与掩码处理
    h = reshape( h, size(h,1), []);
    if isempty(mask)
        mask = ones(size(h));
    else
        mask = reshape( mask, 1, []);
    end
    mask = (sum(h,1) == 0) | mask;
    mask = ~mask;
    
    % ADMM 惩罚参数 (可根据经验调整)
    rho1 = 1.0;
    rho2 = 1.0;
    rho3 = 1.0;

    % 定义核心算子
    ProxF = @(v, vv, lambda_prox) solve_data_prox(lambda_residual, lambda_prox, v, mask, vv, warmstart, ...
                                         h, N, xt, gm, QE, I, amb, DC, scz, gpu );
    ProxG1 = @(v, lambda_prox) proxShrink(v, lambda_prox); % L1 prox for Dx-w
    ProxG2 = @(v, lambda_prox) proxShrink(v, lambda_prox); % L1 prox for Ew

    % 定义线性算子
    D_op = @(x) KMat(x, [1,1], 1);      % Gradient operator
    DT_op = @(x) KMat(x, [1,1], -1);     % Divergence operator
    E_op = @(w) EMat(w, [1,1], 1);      % Symmetrized gradient operator
    ET_op = @(y) EMat(y, [1,1], -1);     % Adjoint of E

    % --- 预计算 x-update 和 w-update 的求解器 ---
    % (rho3*I + rho1*D'D)x = RHS_x
    % (rho1*I + rho2*E'E)w = RHS_w
    % 这两个矩阵是固定的，可以预分解以加速求解
    
    [H, W, C] = size(x0);
    
    % 使用快速傅里叶变换 (FFT) 进行高效求解
    % 这是处理卷积型算子 (如差分算子) 的标准高效方法
    size_x = [H, W];
    otfD_x = psf2otf([1, -1], size_x);
    otfD_y = psf2otf([1; -1], size_x);
    
    Denom_x = rho3 + rho1 * (abs(otfD_x).^2 + abs(otfD_y).^2);
    
    % 对于 w-update, 矩阵更复杂, 但也可以用 FFT 对角化
    % E'E 算子在傅里叶域的表示
    otfE_xx = psf2otf([1, -2, 1], size_x);
    otfE_yy = psf2otf([1; -2; 1], size_x);
    otfE_xy = psf2otf([1, -1; -1, 1], size_x);
    
    % Denom_w 是一个 2x2 的矩阵场，这里为简化，使用共轭梯度法 (CG)
    % CG 对于正定系统非常高效，且无需显式构造矩阵
    cg_tol = 1e-4;
    cg_max_it = 20;
    
    % 初始化 ADMM 变量
    x = x0;
    w = zeros(H, W, C, 2); % 辅助梯度场
    y1 = zeros(H, W, C, 2); % 分裂变量 for Dx-w
    y2 = zeros(H, W, C, 4); % 分裂变量 for Ew
    y3 = x0;                % 分裂变量 for x in data term
    
    u1 = zeros(size(y1)); % 对偶变量
    u2 = zeros(size(y2));
    u3 = zeros(size(y3));

    fprintf('开始 ADMM-TGV 重建...\n');
    
    % --- 2. ADMM 主循环 ---
    for k = 1:max_it
        
        % --- a. x-update ---
        % 求解 (rho3*I + rho1*D'D)x = rho3*(y3 - u3) + rho1*DT*(y1 - u1 + w)
        RHS_x_fft = rho3 * fft2(y3 - u3) + rho1 * DT_op_fft(fft2(y1 - u1 + w), otfD_x, otfD_y);
        x_fft = RHS_x_fft ./ repmat(Denom_x, [1,1,C]);
        x = real(ifft2(x_fft));
        
        % --- b. w-update ---
        % 求解 (rho1*I + rho2*E'E)w = rho1*(D*x - y1 + u1)
        RHS_w = rho1 * (D_op(x) - y1 + u1);
        A_w = @(w_vec) reshape(rho1*reshape(w_vec, size(w)) + rho2*ET_op(E_op(reshape(w_vec, size(w)))), [], 1);
        [w_vec, ~] = pcg(A_w, RHS_w(:), cg_tol, cg_max_it, [], [], w(:));
        w = reshape(w_vec, size(w));
        
        % --- c. y1-update (soft thresholding) ---
        v1 = D_op(x) - w + u1;
        lambda1 = repmat(reshape(alpha1_weights, [1,1,C]), [H, W, 1, 2]) / rho1;
        y1 = ProxG1(v1, lambda1);
        
        % --- d. y2-update (soft thresholding) ---
        v2 = E_op(w) + u2;
        lambda2 = repmat(reshape(alpha0_weights, [1,1,C]), [H, W, 1, 4]) / rho2;
        y2 = ProxG2(v2, lambda2);
        
        % --- e. y3-update (data term prox) ---
        v3 = x + u3;
        y3 = ProxF(v3, y3, 1/rho3); % vv=y3 for warmstart
        
        % --- f. Dual updates ---
        u1 = u1 + (D_op(x) - w - y1);
        u2 = u2 + (E_op(w) - y2);
        u3 = u3 + (x - y3);
        
        % 打印进度 (可选)
        if mod(k, 10) == 0
            fprintf('Iter %d\n', k);
        end
    end
    
    res = y3; % 最终结果是数据项的解
    
    % 恢复尺度
    res(:,:,1) = res(:,:,1)*scz;

return; % --- END of admm_tgv_solver ---

% --- 辅助函数 ---

function prox = proxShrink(v, lambda_prox)
    % Anisotropic soft-thresholding
    prox = max(0, v - lambda_prox) - max(0, -v - lambda_prox);
return;

function Dt_y_fft = DT_op_fft(y_fft, otfD_x, otfD_y)
    % 在傅里叶域计算 D' * y
    y_fft_x = y_fft(:,:,:,1);
    y_fft_y = y_fft(:,:,:,2);
    Dt_y_fft = conj(otfD_x) .* y_fft_x + conj(otfD_y) .* y_fft_y;
return;

function [ result ] = KMat( x, ~, flag ) % weights are handled outside now
    if flag > 0       
        xx = x(:,[2:end end],:)-x;
        xy = x([2:end end],:,:)-x;
        result = cat(4, xx, xy);
    elseif flag < 0 
        xx = x(:,:,:,1)-x(:,[1 1:end-1],:,1);
        xx(:,1,:)   = x(:,1,:,1);
        xx(:,end,:) = -x(:,end-1,:,1);
        xy = x(:,:,:,2)-x([1 1:end-1],:,:,2);
        xy(1,:,:)   = x(1,:,:,2);
        xy(end,:,:) = -x(end-1,:,:,2);
        result = - (xy + xx);
    end
return;

function [ result ] = EMat( w, ~, flag ) % weights are handled outside now
    % w is H x W x C x 2
    w_x = w(:,:,:,1);
    w_y = w(:,:,:,2);
    if flag > 0 % Computes E*w (symmetrized gradient)
        % Returns H x W x C x 4 tensor [w_xx, w_yy, w_xy, w_yx]
        w_xx = w_x(:,[2:end end],:) - w_x;
        w_yy = w_y([2:end end],:,:) - w_y;
        w_xy = w_x([2:end end],:,:) - w_x;
        w_yx = w_y(:,[2:end end],:) - w_y;
        result = cat(4, w_xx, w_yy, w_xy, w_yx);
    elseif flag < 0 % Computes E'*y (adjoint of E)
        % y is H x W x C x 4
        y_xx = w(:,:,:,1); y_yy = w(:,:,:,2);
        y_xy = w(:,:,:,3); y_yx = w(:,:,:,4);
        
        div_x = y_xx - y_xx(:,[1 1:end-1],:);
        div_x(:,1,:) = y_xx(:,1,:); div_x(:,end,:) = -y_xx(:,end-1,:);
        
        div_y = y_yy - y_yy([1 1:end-1],:,:);
        div_y(1,:,:) = y_yy(1,:,:); div_y(end,:,:) = -y_yy(end-1,:,:);
        
        div_xy = y_xy - y_xy([1 1:end-1],:,:);
        div_xy(1,:,:) = y_xy(1,:,:); div_xy(end,:,:) = -y_xy(end-1,:,:);
        
        div_yx = y_yx - y_yx(:,[1 1:end-1],:);
        div_yx(:,1,:) = y_yx(:,1,:); div_yx(:,end,:) = -y_yx(:,end-1,:);
        
        result_x = div_x + div_xy;
        result_y = div_y + div_yx;
        result = cat(4, result_x, result_y);
    end
return;

% NOTE: You need to copy the functions `solve_data_prox` and its dependencies
% (`newton_opt_core_serial`, `obj_grad_func`, `model_func`) from your
% previous implementation into this file. They are reused here without change.

end % End of main function admm_tgv_solver
