function hist_cube = get2Dpx(A, Z, b, gt, N_shots, T)
    % GET2DPX 向量化生成 2D SPAD Pile-up 场景直方图
    %
    % 输入:
    %   A: [M, N] 反射率矩阵
    %   Z: [M, N] 深度矩阵 (Bin单位)
    %   b: 标量, 背景光到达率 (per bin)
    %   gt: 脉冲分布对象 (makedist)
    %   N_shots: 激光发射次数
    %   T: 直方图长度 (Bin数量)
    %
    % 输出:
    %   hist_cube: [M, N, T] 观测直方图数据立方体

    [M, N] = size(A);
    num_pixels = M * N;
    
    % 展平输入以便批处理
    A_flat = A(:);
    Z_flat = Z(:);
    
    % 预分配输出立方体 (展平状态: Pixel x Time)
    hist_flat = zeros(num_pixels, T);
    
    % --- 内存管理策略 ---
    % 计算单次批处理的最大像素数
    % 内存占用估算: Batch * T * N_shots * 8 bytes (double)
    % 设定目标内存占用约 200MB，防止频繁 Swap
    mem_limit_bytes = 200 * 1024 * 1024; 
    bytes_per_element = 8; 
    batch_size = floor(mem_limit_bytes / (T * N_shots * bytes_per_element));
    batch_size = max(batch_size, 1); % 至少处理1个
    
    % 时间轴向量 (1 x T)
    t_vec = 1:T;
    
    fprintf('开始向量化仿真: %d 像素, BatchSize = %d...\n', num_pixels, batch_size);
    
    % --- 批处理循环 ---
    for start_idx = 1:batch_size:num_pixels
        end_idx = min(start_idx + batch_size - 1, num_pixels);
        current_batch_len = end_idx - start_idx + 1;
        
        % 1. 提取当前批次的参数
        a_batch = A_flat(start_idx:end_idx); % [Batch x 1]
        z_batch = Z_flat(start_idx:end_idx); % [Batch x 1]
        
        % 2. 向量化计算潜在到达率 Lambda [Batch x T]
        % 利用隐式扩展: (Batch x 1) - (1 x T) -> (Batch x T)
        % 注意：pdf 函数通常支持向量输入
        % t_grid: [Batch x T]
        t_grid = repmat(t_vec, current_batch_len, 1);
        z_grid = repmat(z_batch, 1, T);
        
        % 计算 PDF 值
        pdf_vals = pdf(gt, t_grid - z_grid);
        pdf_vals(isnan(pdf_vals)) = 0;
        
        % Lambda = A * PDF + b
        lambda_batch = a_batch .* pdf_vals + b;
        
        % 3. 生成光子事件 [Batch x T x N_shots]
        % 扩展 Lambda 到 N_shots 维度
        % 为了节省内存，直接在 poissrnd 中指定维度，而不是 repmat
        % poissrnd 需要 lambda 匹配或者标量，这里我们需要扩展 lambda
        % 技巧：先生成 Poisson 随机数
        
        % 由于 lambda 对于所有 shots 是一样的，我们可以利用 MATLAB 的 repmat 隐式特性
        % 但 poissrnd(Matrix, [M, N]) 这种调用方式不支持第三维扩展 lambda
        % 所以必须显式 repmat lambda，或者循环 shots (太慢)。
        % 鉴于我们已经限制了 batch_size，这里直接 repmat 是安全的。
        lambda_3d = repmat(lambda_batch, [1, 1, N_shots]);
        
        % 生成泊松计数 (Batch x T x Shots)
        counts = poissrnd(lambda_3d);
        
        % 二值化 (SPAD 响应)
        is_event = counts >= 1;
        
        % 4. 模拟 Pile-up (Winner-Takes-All)
        % 在时间维度 (Dim 2) 寻找第一个非零值
        % max 返回的索引即为第一个触发的 bin
        [has_trigger, first_bin_idx] = max(is_event, [], 2);
        
        % has_trigger: [Batch x 1 x Shots] (逻辑值，表示该次脉冲是否探测到光子)
        % first_bin_idx: [Batch x 1 x Shots] (索引值 1~T)
        
        % 压缩维度 -> [Batch x Shots]
        has_trigger = squeeze(has_trigger);
        first_bin_idx = squeeze(first_bin_idx);
        
        % 5. 统计直方图
        % 我们需要将 (pixel_local_idx, bin_idx) 的出现次数累加
        % 仅处理触发了事件的脉冲
        
        if current_batch_len == 1
            % 特殊处理单像素 batch (squeeze 会把维度压扁)
            valid_bins = first_bin_idx(has_trigger);
            % 累加到当前像素直方图
            for k = 1:length(valid_bins)
                b_idx = valid_bins(k);
                hist_flat(start_idx, b_idx) = hist_flat(start_idx, b_idx) + 1;
            end
        else
            % 多像素 batch
            % 找到所有有效触发的线性索引
            % 行索引 (Pixel in Batch): 1..Batch
            % 列索引 (Shots): 1..Shots (不重要，我们只关心总数)
            
            [r_idx, ~] = find(has_trigger); % r_idx 是 batch 内的像素索引
            
            % 获取对应的 bin 索引
            % 注意：first_bin_idx(has_trigger) 提取出的顺序与 find(has_trigger) 一致
            bin_vals = first_bin_idx(has_trigger);
            
            % 使用 linear indexing 快速累加
            % 目标矩阵: hist_flat 的局部块 [Batch x T]
            % 线性索引 = (bin_vals - 1) * Batch + r_idx
            % 但我们需要累加到全局 hist_flat
            
            % 更简单的方法：对当前 batch 建立临时直方图
            % 使用 accumarray: [Pixel_Idx, Bin_Idx] -> Count
            if ~isempty(r_idx)
                local_hist = accumarray([r_idx, bin_vals], 1, [current_batch_len, T]);
                hist_flat(start_idx:end_idx, :) = local_hist;
            end
        end
    end
    
    % 还原 3D 形状
    hist_cube = reshape(hist_flat, [M, N, T]);
end