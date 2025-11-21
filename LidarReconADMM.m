classdef LidarReconADMM
    % LidarReconADMM 激光雷达 Pile-up 反演求解器
    % 算法: Linearized ADMM + Second-order Total Variation (Hessian) Regularization
    % 子问题求解: Explicit Analytic Newton Method (Proximal MLE)
    
    methods(Static)
        function [Z_est, A_est, history] = reconstruct(hist_cube, pulse_params, N_shots, opts)
            % RECONSTRUCT 主求解接口
            % 输入:
            %   hist_cube: [M, N, T] 观测直方图数据立方体
            %   pulse_params: 结构体 {mu, lambda}
            %   N_shots: 激光发射次数
            %   opts: 配置参数 (可选)
            
            arguments
                hist_cube double
                pulse_params struct
                N_shots double
                opts.rho (1,1) double = 1.0       % ADMM 惩罚参数
                opts.gamma_z (1,1) double = 0.5   % 深度正则化权重
                opts.gamma_a (1,1) double = 0.1   % 反射率正则化权重
                opts.max_iter (1,1) double = 30   % ADMM 最大迭代
                opts.mu_ratio (1,1) double = 100  % 线性化参数 mu = rho * ratio (需 > ||K||^2)
                opts.tol (1,1) double = 1e-4      % 收敛阈值
            end
            
            [M, N, T] = size(hist_cube);
            
            % --- 1. 初始化 ---
            % 简单的初始化：Z取峰值，A取均值
            [~, max_idx] = max(hist_cube, [], 3);
            Z = double(max_idx) - pulse_params.mu;
            A = ones(M, N) * 0.5;
            
            % 辅助变量 Y (Y_z_x, Y_z_y, Y_a_x, Y_a_y) - 二阶差分
            % 维度说明: Yz{1}是Z的水平二阶差分, Yz{2}是垂直
            Yz = {zeros(M,N), zeros(M,N)}; 
            Ya = {zeros(M,N), zeros(M,N)};
            
            % 对偶变量 U
            Uz = {zeros(M,N), zeros(M,N)};
            Ua = {zeros(M,N), zeros(M,N)};
            
            % 预计算常数
            rho = opts.rho;
            mu_prox = rho * opts.mu_ratio; % 线性化参数，必须足够大以保证收敛
            
            history.dual_res = [];
            history.primal_res = [];
            
            fprintf('开始 ADMM 重建 (M=%d, N=%d, T=%d)...\n', M, N, T);
            t_start = tic;
            
            % --- 2. ADMM 主循环 ---
            for k = 1:opts.max_iter
                Z_old = Z; A_old = A;
                
                % === Step A: 构造先验引导 (Prior Guidance) ===
                % 计算 K^T(KX - Y + U)
                % K 是二阶差分算子
                
                % 1. 计算当前 KX
                [KZ_x, KZ_y] = LidarReconADMM.op_K(Z);
                [KA_x, KA_y] = LidarReconADMM.op_K(A);
                
                % 2. 计算残差 R = KX - Y + U
                Rz_x = KZ_x - Yz{1} + Uz{1};
                Rz_y = KZ_y - Yz{2} + Uz{2};
                Ra_x = KA_x - Ya{1} + Ua{1};
                Ra_y = KA_y - Ya{2} + Ua{2};
                
                % 3. 计算 K^T(R) (伴随算子)
                KtR_z = LidarReconADMM.op_Kt(Rz_x, Rz_y);
                KtR_a = LidarReconADMM.op_Kt(Ra_x, Ra_y);
                
                % 4. 计算 Proximal 目标锚点 (Target Anchor)
                % X_tilde = X_k - (rho/mu) * K^T(...)
                Z_tilde = Z - (rho / mu_prox) * KtR_z;
                A_tilde = A - (rho / mu_prox) * KtR_a;
                
                % === Step B: 并行投影 (The x-update) ===
                % 求解单像素 Proximal MLE: min NLL + (mu/2)||x - x_tilde||^2
                % 使用 parfor 加速 (如果 M*N 较大)
                
                % 展平数据以适应 parfor
                vhist_flat = reshape(hist_cube, M*N, T);
                z_tilde_flat = Z_tilde(:);
                a_tilde_flat = A_tilde(:);
                z_new_flat = zeros(M*N, 1);
                a_new_flat = zeros(M*N, 1);
                
                parfor i = 1:(M*N)
                    % 提取单像素数据
                    h_i = vhist_flat(i, :)';
                    if sum(h_i) < 1 % 跳过无效像素
                        z_new_flat(i) = z_tilde_flat(i);
                        a_new_flat(i) = a_tilde_flat(i);
                        continue; 
                    end
                    
                    % 构造初值 (利用上一帧结果)
                    init_guess = [z_tilde_flat(i); a_tilde_flat(i)];
                    
                    % 调用极速求解器 (Proximal 版本)
                    [est_z, est_a] = LidarReconADMM.solve_one_pixel_prox(...
                        h_i, N_shots, pulse_params, ...
                        init_guess, mu_prox, z_tilde_flat(i), a_tilde_flat(i));
                    
                    z_new_flat(i) = est_z;
                    a_new_flat(i) = est_a;
                end
                
                Z = reshape(z_new_flat, M, N);
                A = reshape(a_new_flat, M, N);
                
                % === Step C: 空间平滑 (The y-update) ===
                % Y = SoftThreshold(KX + U, gamma/rho)
                
                [KZ_x, KZ_y] = LidarReconADMM.op_K(Z);
                [KA_x, KA_y] = LidarReconADMM.op_K(A);
                
                thresh_z = opts.gamma_z / rho;
                thresh_a = opts.gamma_a / rho;
                
                Yz{1} = LidarReconADMM.soft_thresh(KZ_x + Uz{1}, thresh_z);
                Yz{2} = LidarReconADMM.soft_thresh(KZ_y + Uz{2}, thresh_z);
                Ya{1} = LidarReconADMM.soft_thresh(KA_x + Ua{1}, thresh_a);
                Ya{2} = LidarReconADMM.soft_thresh(KA_y + Ua{2}, thresh_a);
                
                % === Step D: 对偶更新 (The u-update) ===
                % U = U + (KX - Y)
                Uz{1} = Uz{1} + (KZ_x - Yz{1});
                Uz{2} = Uz{2} + (KZ_y - Yz{2});
                Ua{1} = Ua{1} + (KA_x - Ya{1});
                Ua{2} = Ua{2} + (KA_y - Ya{2});
                
                % === 收敛监控 ===
                diff_norm = norm(Z - Z_old, 'fro') / norm(Z_old, 'fro');
                history.primal_res(end+1) = diff_norm;
                
                if mod(k, 5) == 0 || k==1
                    fprintf('Iter %2d | Diff: %.2e | Time: %.2fs\n', k, diff_norm, toc(t_start));
                end
                
                if diff_norm < opts.tol
                    fprintf('Converged at iter %d.\n', k);
                    break;
                end
            end
            
            Z_est = Z;
            A_est = A;
        end
        
        % ---------------------------------------------------------
        % 核心子问题求解器: Proximal Newton Method
        % ---------------------------------------------------------
        function [z_out, a_out] = solve_one_pixel_prox(vhist, N, pulse_params, init_guess, mu_prox, z_prior, a_prior)
            % 求解 min NLL(z,a) + (mu/2)*||[z;a] - [z_prior;a_prior]||^2
            % 这是一个强凸问题 (由于 mu_prox 很大)
            
            % 预计算
            T = length(vhist);
            t_axis = (1:T)';
            h_sum = sum(vhist);
            h0 = N - h_sum;
            
            % 反向累积和 (手动循环优化)
            S = zeros(T, 1);
            cum_val = 0;
            for i = T:-1:1
                cum_val = cum_val + vhist(i);
                S(i) = cum_val - vhist(i);
            end
            
            % 波形常数
            mu = pulse_params.mu;
            lam = pulse_params.lambda;
            const_log = 0.5 * log(lam / (2*pi));
            const_deriv = lam / (2*mu^2);
            mu_sq = mu^2;
            
            % 变量
            z = init_guess(1);
            a = init_guess(2);
            b = 0.001; % 简化：假设背景光很小且固定，或者作为第3个参数优化(此处为速度仅优化z,a)
            
            % 优化循环
            for iter = 1:10 % Proximal Newton 收敛极快，10次足矣
                % 1. 波形计算
                dt = t_axis - z;
                mask = dt > 1e-9;
                
                g_val = zeros(T, 1);
                g_prime = zeros(T, 1);
                
                if any(mask)
                    xm = dt(mask);
                    inv_xm = 1 ./ xm;
                    xm_sq = xm .* xm;
                    term_exp = -const_deriv .* (xm - mu).^2 .* inv_xm;
                    val_m = exp(const_log - 1.5 * log(xm) + term_exp);
                    g_val(mask) = val_m;
                    term_deriv = -1.5 .* inv_xm - const_deriv .* (1 - mu_sq ./ xm_sq);
                    g_prime(mask) = val_m .* term_deriv;
                end
                
                % 2. 梯度与 Hessian (NLL部分)
                lambda = a .* g_val + b;
                lambda = max(lambda, 1e-10);
                
                exp_neg_lbd = exp(-lambda);
                denom = 1 - exp_neg_lbd;
                inv_denom = 1 ./ denom;
                w = vhist .* exp_neg_lbd .* inv_denom;
                u = w .* inv_denom;
                
                dL_dlbd = (h0 + S) - w;
                
                J_z = -a .* g_prime;
                J_a = g_val;
                
                % 标量化计算 Hessian 元素
                u_Jz = u .* J_z;
                u_Ja = u .* J_a;
                
                Hzz = sum(u_Jz .* J_z);
                Hza = sum(u_Jz .* J_a);
                Haa = sum(u_Ja .* J_a);
                
                gz = sum(dL_dlbd .* J_z);
                ga = sum(dL_dlbd .* J_a);
                
                % 3. 加入 Proximal 项 (正则化)
                % Loss += (mu/2) * ((z - z_prior)^2 + (a - a_prior)^2)
                % Grad += mu * (x - x_prior)
                % Hess += mu * I
                
                gz = gz + mu_prox * (z - z_prior);
                ga = ga + mu_prox * (a - a_prior);
                
                Hzz = Hzz + mu_prox;
                Haa = Haa + mu_prox;
                
                % 4. 显式求解 (2x2 Cramer's Rule)
                det_H = Hzz * Haa - Hza * Hza;
                % 这里的 det_H 肯定 > 0，因为 H_NLL 半正定且 mu_prox > 0
                inv_det = 1 / det_H;
                
                dz = inv_det * (Haa * (-gz) - Hza * (-ga));
                da = inv_det * (-Hza * (-gz) + Hzz * (-ga));
                
                % 5. 更新
                z = z + dz;
                a = a + da;
                
                % 简单投影
                a = max(a, 0);
                z = max(min(z, T+5), -5);
                
                if (dz^2 + da^2) < 1e-12
                    break;
                end
            end
            z_out = z;
            a_out = a;
        end
        
        % ---------------------------------------------------------
        % 算子定义 (Operators)
        % ---------------------------------------------------------
        function [Gx, Gy] = op_K(X)
            % 二阶差分算子 (Second-order Difference)
            % 对应 Hessian 正则化
            Gx = conv2(X, [1, -2, 1], 'same');
            Gy = conv2(X, [1; -2; 1], 'same');
        end
        
        function X = op_Kt(Gx, Gy)
            % 二阶差分算子的伴随 (Adjoint)
            % 卷积核翻转后相同 (对称)
            % 注意：'same' 模式下的边界处理在反演中通常由数据项主导
            X = conv2(Gx, [1, -2, 1], 'same') + conv2(Gy, [1; -2; 1], 'same');
        end
        
        function Y = soft_thresh(X, T)
            % 软阈值算子
            Y = sign(X) .* max(abs(X) - T, 0);
        end
    end
end