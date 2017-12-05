clear; clc;
format compact
format short

%% benchmark univariate polynomial 
batch_size = 10;
max_deg = 2;
coeff_lb = -1e3;
coeff_ub = 1e3;
var_lb = -1e6;
var_ub = 1e6;
tol_x = 1e-6;
options = optimset('TolX', tol_x);
precision_x = tol_x;
precision_y = 1e-6;
benchmark_up(batch_size, max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

max_deg = 3;
var_lb = -1e5;
var_ub = 1e5;
tol_x = 1e-4;
options = optimset('TolX', tol_x);
precision_x = tol_x;
benchmark_up(batch_size, max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

max_deg = 5;
var_lb = -1e3;
var_ub = 1e3;
benchmark_up(batch_size, max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

max_deg = 10;
var_lb = -1e2;
var_ub = 1e2;
benchmark_up(batch_size, max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

max_deg = 30;
var_lb = -1;
var_ub = 1;
benchmark_up(batch_size, max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

%% benchmark multivariate polynomial 
% batch_size = 500;
% num_var = 2;
% max_deg = 10;
% max_term = 2;
% coeff_lb = -1e2;
% coeff_ub = 1e2;
% var_lb = -1e3;
% var_ub = 1e3;
% tol_x = 1e-6;
% options = optimoptions('fmincon', 'TolX', tol_x);
% up_options = optimset(optimset('fminbnd'), 'TolX', tol_x);
% precision_x = tol_x;
% precision_y = 1e-6;
% benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, up_options, precision_x, precision_y);
% 
% num_var = 5;
% max_deg = 15;
% max_term = 10;
% var_lb = -1e3;
% var_ub = 1e3;
% benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, up_options, precision_x, precision_y);
% 
% num_var = 10;
% max_deg = 20;
% max_term = 20;
% var_lb = -1e2;
% var_ub = 1e2;
% benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, up_options, precision_x, precision_y);
% 
% batch_size = 25;
% num_var = 30;
% max_deg = 60;
% max_term = 30;
% var_lb = -1;
% var_ub = 1;
% benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, up_options, precision_x, precision_y);

%% benchmark sos
% batch_size = 1;
% num_var = 2;
% max_deg = 10;
% max_term = 2;
% coeff_lb = -1e2;
% coeff_ub = 1e2;
% var_lb = -inf;
% var_ub = +inf;
% tol_x = 1e-6;
% options = optimoptions('fmincon', 'TolX', tol_x);
% up_options = optimset(optimset('fminbnd'), 'TolX', tol_x);
% precision_x = tol_x;
% precision_y = 1e-6;
% benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, up_options, precision_x, precision_y);

% num_var = 5;
% max_deg = 15;
% max_term = 10;
% var_lb = -1e3;
% var_ub = 1e3;
% benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, up_options, precision_x, precision_y);

% num_var = 10;
% max_deg = 20;
% max_term = 20;
% var_lb = -1e2;
% var_ub = 1e2;
% benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, up_options, precision_x, precision_y);

% batch_size = 25;
% num_var = 30;
% max_deg = 60;
% max_term = 30;
% var_lb = -1;
% var_ub = 1;
% benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, up_options, precision_x, precision_y);

function tb = benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, up_options, precision_x, precision_y)
    fprintf('\nMultivariate Polynomial\n--------------------------------------------------------\n');
    disp(['batch size = ', num2str(batch_size)]);
    disp(['number of variables = ', num2str(num_var)]);
    disp(['max degree = ', num2str(max_deg)]);
    disp(['number of terms ', num2str(max_term)]);
    disp(['range of coefficient = [', num2str(coeff_lb), ', ', num2str(coeff_ub), ']']);
    disp(['range of x(i) = [', num2str(var_lb), ', ', num2str(var_ub), ']']);
    disp(['termination tolerance on x = ', num2str(options.TolX)]);
    disp(['precision for comparing x = ', num2str(precision_x)]);
    disp(['precision for comparing f(x) = ', num2str(precision_y * 100), '%']);
    
    con = 'fmincon';
    for i = 1:num_var
        sym_vars(i) = sym(strcat('sv', num2str(i)));
    end

    time_con = 0;
    time_ip = 0;
    time_sqp = 0;
    time_as = 0;
    time_trr = 0;
    time_gsccd_nlp = 0;
    time_gsccd_root = 0;
    time_sos = 0;

    ip_eq_nlp = 0;
    ip_beat_nlp = 0;
    ip_lose_nlp = 0;

    sqp_eq_nlp = 0;
    sqp_beat_nlp = 0;
    sqp_lose_nlp = 0;

    trr_eq_nlp = 0;
    trr_beat_nlp = 0;
    trr_lose_nlp = 0;

    root_eq_nlp = 0;
    root_beat_nlp = 0;
    root_lose_nlp = 0;

    sos_eq_nlp = 0;
    sos_beat_nlp = 0;
    sos_lose_nlp = 0;


    for i = 1:batch_size
        p = rand_mp_prob(num_var, max_deg, max_term, coeff_ub, coeff_lb, var_lb, var_ub, options);
        f = p.objective;
        f_symfun = f(sym_vars);
        g_symfun = gradient(f_symfun, sym_vars);
        h_symfun = hessian(f_symfun, sym_vars);
        g = matlabFunction(g_symfun, 'Vars', {sym_vars});
        h = matlabFunction(h_symfun, 'Vars', {sym_vars});
        
        % t = clock;
        % [x_con,y_con] = mp_solve(p, con);
        % time_con = time_con + etime(clock, t);
        
        p.options.Algorithm = 'interior-point';
        p.objective = include_g_h(f, g, h);
        p.options.SpecifyObjectiveGradient = true;
        p.options.HessianFcn = gen_hessianfcn(f, h);
        t = clock;
        [x_ip, y_ip] = mp_solve(p, con);
        time_ip = time_ip + etime(clock, t);
        
        p.options.Algorithm = 'trust-region-reflective';
        t = clock;
        [x_trr, y_trr] = mp_solve(p, con);
        time_trr = time_trr + etime(clock, t);
        p.objective = f;
        p.options.SpecifyObjectiveGradient = false;
        p.options.HessianFcn = [];
        
        p.options.Algorithm = 'sqp';
        t = clock;
        [x_sqp, y_sqp] = mp_solve(p, con);
        time_sqp = time_sqp + etime(clock, t);
        
        % p.options.Algorithm = 'active-set';
        % t = clock;
        % [x_as, y_as] = mp_solve(p, con);
        % time_as = time_as + etime(clock, t);
        
        t = clock;
        [x_gsccd_nlp, y_gsccd_nlp] = mp_solve(p, 'gsccd', 'nlp', up_options);
        time_gsccd_nlp = time_gsccd_nlp + etime(clock, t);
        
        t = clock;
        [x_gsccd_root, y_gsccd_root] = mp_solve(p, 'gsccd', 'root', []);
        time_gsccd_root = time_gsccd_root + etime(clock, t);

        % t = clock;
        % [~, y_sos] = mp_solve(p, 'sos');
        % time_sos = time_sos + etime(clock, t);
        % y_sos

        xs = [x_ip; x_sqp; x_trr; x_gsccd_nlp; x_gsccd_root];
        for j = 1:size(xs,1)
            for k = 1:num_var
                assert((xs(j,k) - p.lb(k) >= -options.TolX && xs(j,k) - p.ub(k) <= options.TolX));
            end
        end
        
        % compare objective value
        ip_eq_nlp = ip_eq_nlp + is_eq(x_ip, x_gsccd_nlp, precision_x, y_ip, y_gsccd_nlp, precision_y);
        ip_beat_nlp = ip_beat_nlp + is_better(x_ip, x_gsccd_nlp, precision_x, y_ip, y_gsccd_nlp, precision_y);
        ip_lose_nlp = ip_lose_nlp + is_worse(x_ip, x_gsccd_nlp, precision_x, y_ip, y_gsccd_nlp, precision_y);

        sqp_eq_nlp = sqp_eq_nlp + is_eq(x_sqp, x_gsccd_nlp, precision_x, y_sqp, y_gsccd_nlp, precision_y);
        sqp_beat_nlp = sqp_beat_nlp + is_better(x_sqp, x_gsccd_nlp, precision_x, y_sqp, y_gsccd_nlp, precision_y);
        sqp_lose_nlp = sqp_lose_nlp + is_worse(x_sqp, x_gsccd_nlp, precision_x, y_sqp, y_gsccd_nlp, precision_y);

        trr_eq_nlp = trr_eq_nlp + is_eq(x_trr, x_gsccd_nlp, precision_x, y_trr, y_gsccd_nlp, precision_y);
        trr_beat_nlp = trr_beat_nlp + is_better(x_trr, x_gsccd_nlp, precision_x, y_trr, y_gsccd_nlp, precision_y);
        trr_lose_nlp = trr_lose_nlp + is_worse(x_trr, x_gsccd_nlp, precision_x, y_trr, y_gsccd_nlp, precision_y);

        root_eq_nlp = root_eq_nlp + is_eq(x_gsccd_root, x_gsccd_nlp, precision_x, y_gsccd_root, y_gsccd_nlp, precision_y);
        root_beat_nlp = root_beat_nlp + is_better(x_gsccd_root, x_gsccd_nlp, precision_x, y_gsccd_root, y_gsccd_nlp, precision_y);
        root_lose_nlp = root_lose_nlp + is_worse(x_gsccd_root, x_gsccd_nlp, precision_x, y_gsccd_root, y_gsccd_nlp, precision_y);
    end

    beat_nlp = [ip_beat_nlp; sqp_beat_nlp; trr_beat_nlp; root_beat_nlp] / batch_size;
    lose_nlp = [ip_lose_nlp; sqp_lose_nlp; trr_lose_nlp; root_lose_nlp] / batch_size;
    eq_nlp = [ip_eq_nlp; sqp_eq_nlp; trr_eq_nlp; root_eq_nlp] / batch_size;
    fprintf('\n');
    disp(table([time_ip], [time_sqp], [time_trr], [time_gsccd_nlp], [time_gsccd_root], 'VariableNames', {'ip', 'sqp' ,'trr', 'gsccd_nlp', 'gsccd_root'}, 'RowNames', {'time'}));
    fprintf('\n');
    disp(table(beat_nlp, lose_nlp, eq_nlp, 'VariableNames', {'beat_nlp', 'lose_nlp', 'eq_nlp'}, 'RowNames', {'ip', 'sqp', 'trr', 'gsscd_root'}));
    fprintf('\n');
end

function benchmark_up(batch_size, max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y)
    fprintf('\nUnivariate Polynomial\n--------------------------------------------------------\n');
    disp(['batch size = ', num2str(batch_size)]);
    disp(['max degree = ', num2str(max_deg)]);
    disp(['range of coefficient = [', num2str(coeff_lb), ', ', num2str(coeff_ub), ']']);
    disp(['range of x = [', num2str(var_lb), ', ', num2str(var_ub), ']']);
    disp(['termination tolerance on x = ', num2str(options.TolX)]);
    disp(['precision for comparing x = ', num2str(precision_x)]);
    disp(['precision for comparing f(x) = ', num2str(precision_y * 100), '%']);
    
    time_nlp = 0;
    time_root = 0;
    time_sos = 0;
    
    nlp_eq_root = 0;
    nlp_beat_root = 0;
    nlp_lose_root = 0;
    
    nlp_eq_sos = 0;
    nlp_beat_sos = 0;
    nlp_lose_sos = 0;
    
    root_eq_sos = 0;
    root_beat_sos = 0;
    root_lose_sos = 0;
    
    for i = 1:batch_size
        p = rand_up_prob(max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options);
        f = p.objective;

        t = clock;
        [x_nlp, y_nlp] = up_solve(p, 'nlp');
        time_nlp = time_nlp + etime(clock, t);
        
        t = clock;
        [x_root, y_root] = up_solve(p, 'root');
        time_root = time_root + etime(clock, t);
        
        t = clock;
        [~, y_sos] = up_solve(p, 'sos');
        time_sos = time_sos + etime(clock, t);
        
        xs = [x_nlp; x_root];
        for j = 1:size(xs,1)
            assert((xs(j) >= p.x1 && xs(j) <= p.x2));
        end

        nlp_eq_root = nlp_eq_root + is_eq(x_nlp, x_root, precision_x, y_nlp, y_root, precision_y);
        nlp_beat_root = nlp_beat_root + is_better(x_nlp, x_root, precision_x, y_nlp, y_root, precision_y);
        nlp_lose_root = nlp_lose_root + is_worse(x_nlp, x_root, precision_x, y_nlp, y_root, precision_y);

        nlp_eq_sos = nlp_eq_sos + is_eq(0, 1, precision_x, y_nlp, y_sos, precision_y);
        nlp_beat_sos = nlp_beat_sos + is_better(0, 1, precision_x, y_nlp, y_sos, precision_y);
        nlp_lose_sos = nlp_lose_sos + is_worse(0, 1, precision_x, y_nlp, y_sos, precision_y);
 
        root_eq_sos = root_eq_sos + is_eq(0, 1, precision_x, y_root, y_sos, precision_y);
        root_beat_sos = root_beat_sos + is_better(0, 1, precision_x, y_root, y_sos, precision_y);
        root_lose_sos = root_lose_sos + is_worse(0, 1, precision_x, y_root, y_sos, precision_y);
    end
    
    fprintf('\n');
    disp(table([time_nlp], [time_root], [time_sos], 'VariableNames', {'fminbnd', 'root', 'sos'}, 'RowNames', {'time'}));
    fprintf('\n');
    disp(table([nlp_beat_root/batch_size], [nlp_lose_root/batch_size], [nlp_eq_root/batch_size], 'VariableNames', {'nlp', 'root', 'equal'}, 'RowNames', {'% win'}));
    fprintf('\n');
    disp(table([nlp_beat_sos/batch_size], [nlp_lose_sos/batch_size], [nlp_eq_sos/batch_size], 'VariableNames', {'nlp', 'sos', 'equal'}, 'RowNames', {'% win'}));
    fprintf('\n');
    disp(table([root_beat_sos/batch_size], [root_lose_sos/batch_size], [root_eq_sos/batch_size], 'VariableNames', {'root', 'sos', 'equal'}, 'RowNames', {'% win'}));
    fprintf('\n');
end

%% multivariate polynomial solver
function [x y] = mp_solve(p, solver, up_solver, up_options)
    if strcmp(solver, 'fmincon')
        [x y] = mp_solve_fmincon(p);
    elseif strcmp(solver, 'gsccd')
        [x y] = mp_solve_gsccd(p, up_solver, up_options);
    elseif strcmp(solver, 'sos')
        [x y] = mp_solve_sos(p);
    else
        error('Unrecognized multivariate polynomial solver');
    end
    x = double(x);
    y = double(y);
end

%% wrapper of fmincon
function [x y] = mp_solve_fmincon(p)
    p.solver = 'fmincon';
    [T x y] = evalc('fmincon(p)');
end

function f_hessian = gen_hessianfcn(f, h)
    function Hout = hessianfcn(x,lambda)
        Hout = h(x'); % no ineq/eq constraint
    end
    f_hessian = @hessianfcn;
end

function f_gradient = include_g_h(f, g, h)
    function [fun, grad, H] = gradientfcn(x)
        fun = f(x);
        if nargout > 1
            grad = g(x);
            if nargout > 2
                H = h(x);
            end
        end
    end
    f_gradient = @gradientfcn;
end

%% Gauss-Seidel cyclic coordinate descent methods
function [x y] = mp_solve_gsccd(p, up_solver, up_options)
    syms variable
    f = p.objective;
    num_var = length(p.x0);
    x = p.x0;
    iter = 0;
    done = java.util.HashSet();
    while (~exist('x_old', 'var') || norm(x - x_old) > p.options.TolX) && iter < 10
        iter = iter + 1;
        x_old = x;
        for i = 1:num_var
            if done.contains(i)
                continue
            end
            x = horzcat(x(1 : i-1), variable, x(i+1 : num_var)); 
            up_f = matlabFunction(f(x)); % create function handle of univariate polynomial up_f = @(y) f(subs(x, variable, y));
            if nargin(up_f) == 0 % skip function taking no arg (e.g. @() 0.0) which means x(i) does not matter
                x(i) = x_old(i);
                continue
            end
            up = struct('objective', up_f, 'x1', p.box(i,1), 'x2', p.box(i,2), 'options', up_options);
            [x_opt y_opt] = up_solve(up, up_solver);
            x(i) = x_opt;
            if x(i) == x_old(i)
                done.add(i);
            end
        end
    end
    y = f(x);
end

%% SOS
function [x y] = mp_solve_sos(p)
    syms gam;
    for i = 1:length(p.x0)
        sym_vars(i) = sym(strcat('sv', num2str(i)));
    end
    vartable = sym_vars;
    prog = sosprogram(vartable);
    prog = sosdecvar(prog,[gam]);
    f = p.sym_obj;
    prog = sosineq(prog,(f-gam));
    % for i = 1:length(p.x0)
    %     prog = sosineq(prog, sym_vars(i) - p.box(i,1));
    %     prog = sosineq(prog, p.box(i,2) - sym_vars(i));
    % end
    prog = sossetobj(prog,-gam);
    [T prog] = evalc('sossolve(prog)');
    x = -inf; % fake minimizer
    y = sosgetsol(prog,gam, 10);
end

%% univariate polynomial solver
function [x y] = up_solve(p, solver)
    if strcmp(solver, 'nlp')
        [x y] = up_solve_nlp(p);
    elseif strcmp(solver, 'root')
        [x y] = up_solve_root(p);
    elseif strcmp(solver, 'sos')
        [x y] = up_solve_sos(p);
    else
        error('Unrecognized univariate polynomial solver');
    end
end

%% univariate polynomial solver by non-linear programming solver
% f: function handle of univariate polynomial
function [x y] = up_solve_nlp(p)
    f = p.objective;
    p.solver = 'fminbnd';
    [local_minimizer, fval] = fminbnd(p);
    [min_val, min_idx] = min([fval; f(p.x1); f(p.x2)]);
    candidates = [local_minimizer p.x1 p.x2];
    x = double(candidates(min_idx));
    y = double(min_val);
end

%% univariate polynomial solver by finding root of derivative
% f: function handle of univariate polynomial
function [x y] = up_solve_root(p)
    f = p.objective;
    syms y
    sym_f = f(y);
    sym_df = diff(sym_f);
    all_roots = roots(sym2poly(sym_df));
    real_roots = all_roots(imag(all_roots) == 0)';
    candidates = horzcat(real_roots, p.x1, p.x2);
    candidates = candidates(and(candidates >= p.x1, candidates <= p.x2));
    min_idx = 1;
    min_val = f(candidates(1));
    for i = 2:length(candidates)
        val = f(candidates(i));
        if (val < min_val)
            min_val = val;
            min_idx = i;
        end
    end
    x = double(candidates(min_idx));
    y = double(min_val);
end

%% SOS
function [xopt y] = up_solve_sos(p)
    syms x gam;
    vartable = [x];
    prog = sosprogram(vartable);
    prog = sosdecvar(prog,[gam]);
    f = p.sym_obj;
    prog = sosineq(prog,(f-gam), [p.x1 p.x2]);
    prog = sossetobj(prog,-gam);
    [T prog] = evalc('sossolve(prog)');
    digit = 10;
    xopt = -inf; % fake optimizer
    y = sosgetsol(prog,gam, digit);
end

%% generate a random multivariate polynomial minimization problem
function prob = rand_mp_prob(num_var, max_deg, max_term, coeff_ub, coeff_lb, var_lb, var_ub, options)
    prob.mp = rand_mulpoly(num_var, max_deg, max_term, coeff_ub, coeff_lb);
    prob.objective = mulpoly2fun(prob.mp);
    for i = 1:num_var
        sym_vars(i) = sym(strcat('sv', num2str(i)));
    end
    prob.sym_obj = prob.objective(sym_vars);
    prob.box = rand_box(num_var, var_lb, var_ub);
    prob.x0 = rand_initpoint(num_var, prob.box);
    % [A b] = box2constraint(prob.box);
    % prob.Aineq = A;
    % prob.bineq = b;
    if var_lb ~= -inf
        prob.lb = prob.box(:,1)';
    end
    if var_ub ~= inf
        prob.ub = prob.box(:,2)';
    end
    prob.options = options;
end

%% generate a random univariate polynomial minimization problem
function prob = rand_up_prob(max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options)
    coeff = rand(1, max_deg+1) * (coeff_ub - coeff_lb) + coeff_lb;
    prob.sym_obj = poly2sym(coeff);
    prob.objective = matlabFunction(prob.sym_obj);
    prob.x1 = rand() * (var_ub - var_lb) + var_lb;
    prob.x2 = rand() * (var_ub - prob.x1) + prob.x1;
    prob.options = options;
end

%% generate a random initial point under given constraint
function x0 = rand_initpoint(num_var, box)
    for i = 1:num_var
        x0(i) = (box(i,2) - box(i,1)) * rand() + box(i,1);
    end
end

%% convert a box into constraint of form Ax <= b
function [A b] = box2constraint(box)
    for i = 1:size(box,1)
        A(2*i-1, i) = -1;
        b(2*i-1, 1) = -box(i, 1);
        A(2*i, i) = 1;
        b(2*i, 1) = box(i, 2);
    end
end

%% generate a random box represented by an nx2 matrix
% i_th row is the interval [a b] which x_i belongs to
function box = rand_box(num_var, var_lb, var_ub)
    for i = 1:num_var
        box(i,1) = (var_ub - var_lb) * rand() + var_lb;
        box(i,2) = (var_ub - box(i,1)) * rand() + box(i,1);
    end
end

%% create function handle of a multivariate polynomial
function fun = mulpoly2fun(mp)
    fun = @(x) mulpoly2fun_helper(x, mp);
end

function result = mulpoly2fun_helper(x, mp)
    result = 0;
    num_terms = length(mp.coeffs);
    for i = 1:num_terms
        m.exp = mp.exps(i,:);
        m.coeff = mp.coeffs(i,1);
        f = monomial2fun(m);
        result = result + f(x);
    end
end

%% create function handle of a monomial
function fun = monomial2fun(m)
    fun = @(x) (m.coeff)*prod(x.^(m.exp));
end

%% generate a multivariate polynomial, which contains
% an exponent matrix where each row represents an exponent
% a vector of coefficients
function mp = rand_mulpoly(num_var, max_deg, max_term, coeff_ub, coeff_lb)
    for i = 1:max_term
        m = rand_monomial(num_var, max_deg, coeff_lb, coeff_ub);
        mp.exps(i,:) = m.exp;
        mp.coeffs(i,1) = m.coeff;
    end
end

%% generate a random monomial, represented by
% an exponent represented by a row vector
% a coefficient
function m = rand_monomial(num_var, max_deg, coeff_lb, coeff_ub)
    expected_e = floor(max_deg / num_var);
    for i = 1:num_var
        if max_deg > 0
            e = floor(randn() * expected_e / 3 + expected_e);
            if e < 0
                e = 0;
            elseif e > max_deg
                e = max_deg;
            end
            m.exp(i) = e;
            max_deg = max_deg - e;
        else
            m.exp(i) = 0;
        end
    end
    m.exp = rand_permute(m.exp);
    m.coeff = double((coeff_ub - coeff_lb) * rand() + coeff_lb);
end

%% permute an array in random order
function b = rand_permute(a)
    order = randperm(length(a));
    for i = 1:length(a)
        b(i) = a(order(i));
    end
end

%% compare result of different optimization methods
function eq = is_eq(x, x_base, precision_x, y, y_base, precision_y)
    eq = norm(x_base - x) <= precision_x || abs(y_base - y) <= abs(y_base * precision_y);
end

function worse = is_worse(x, x_base, precision_x, y, y_base, precision_y)
    worse = ~is_eq(x_base, x, precision_x, y_base, y, precision_y) && y > y_base;
end

function better = is_better(x, x_base, precision_x, y, y_base, precision_y)
    better = ~is_eq(x_base, x, precision_x, y_base, y, precision_y) && y < y_base;
end