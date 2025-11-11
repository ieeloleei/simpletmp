function prox = proxElasticNet(v, lambda_prox, alpha1, alpha2)
    scale_factor = lambda_prox / (lambda_prox + alpha2);
    threshold = alpha1 / (lambda_prox + alpha2);
    v_scaled = scale_factor * v;
    
    % 使用 Isotropic (TV-like) 版本以获得更好的旋转不变性
    Amplitude = sqrt(sum(v_scaled.^2, 4));
    % 增加一个小的 epsilon 防止除以零
    Amplitude(Amplitude == 0) = eps; 
    prox = max(0, 1 - threshold ./ Amplitude) .* v_scaled;
end
