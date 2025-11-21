function [vhist,varr,vlbd] = get1px(a,z,b,gt,N,T)
    % get1px 单像素 SPAD Pile-up 物理仿真
    % 输入:
    %   a: 反射率
    %   z: 深度 (bin)
    %   b: 背景光率
    %   gt: 脉冲分布对象 (makedist)
    %   N: 脉冲次数
    %   T: 直方图长度
    
    % 1. 计算潜在光子到达率
    % pdf 函数支持向量化输入
    t_idx = (1:T)';
    vec_lbd = a * pdf(gt, t_idx - z);
    vec_lbd(isnan(vec_lbd)) = 0;
    
    % 2. 生成泊松光子事件 (包含背景光)
    % 生成 T x N 的矩阵
    lambda_mat = repmat(vec_lbd, 1, N);
    Marr = random('Poisson', lambda_mat) + random('Poisson', b * ones(T, N));
    Marr = (Marr >= 1); % SPAD 二值响应
    
    % 3. 模拟 Pile-up (Winner-Takes-All)
    Mhist = zeros(size(Marr));
    [maxMarr, maxRowInd] = max(Marr, [], 1);
    
    % 仅保留每列第一个有效事件
    valid_cols = maxMarr >= 1;
    valid_rows = maxRowInd(valid_cols);
    cols_idx = find(valid_cols);
    
    % 线性索引赋值
    lin_idx = sub2ind(size(Marr), valid_rows, cols_idx);
    Mhist(lin_idx) = 1;
    
    % 4. 统计直方图
    vhist = sum(Mhist, 2);
    varr = sum(Marr, 2); % 理想无死时间计数
    vlbd = vec_lbd;
end