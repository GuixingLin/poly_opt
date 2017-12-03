clear; clc;
format short

%? different value of b2 affect x_opt????
% f = @(x) 2*x(1)^3 + 2*x(3)*x(1)^2;
% x0 = [-3.3255   -0.6171   -9.9421];
% box = [-4,3;-5,5;9,10];
% [A b] = box2constraint(box);
% A
% b
% x_opt1 = fmincon(f, x0, A, b);
% A2 = zeros(1,num_var)
% b2 = 0
% Aeq2 = zeros(1,num_var);
% beq2 = 0;
% lower = box(:,1)';
% upper = box(:,2)';
% x_opt2 = fmincon(f, x0, A2, b2, Aeq2, beq2, lower, upper);
% x_opt1
% x_opt2
% fmincon(f, x0, ...) % pass explicit upper and lower bound

%% benchmark univariate polynomial 
batch_size = 1000;
max_deg = 2;
coeff_lb = -1e3;
coeff_ub = 1e3;
var_lb = -1e6;
var_ub = 1e6;
tol_x = 1e-6;
options = optimset('TolX', tol_x);
precision_x = tol_x;
precision_y = 1e-6;
% benchmark_up(batch_size, max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

max_deg = 3;
var_lb = -1e5;
var_ub = 1e5;
tol_x = 1e-4;
options = optimset('TolX', tol_x);
precision_x = tol_x;
% benchmark_up(batch_size, max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

max_deg = 5;
var_lb = -1e3;
var_ub = 1e3;
% benchmark_up(batch_size, max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

max_deg = 10;
var_lb = -1e2;
var_ub = 1e2;
% benchmark_up(batch_size, max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

max_deg = 30;
var_lb = -1;
var_ub = 1;
% benchmark_up(batch_size, max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

%% benchmark multivariate polynomial 
batch_size = 10;
num_var = 2;
max_deg = 10;
max_term = 2;
coeff_lb = -1e2;
coeff_ub = 1e2;
var_lb = -1e3;
var_ub = 1e3;
tol_x = 1e-6;
options = optimset('TolX', tol_x);
precision_x = tol_x;
precision_y = 1e-6;
benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

num_var = 5;
max_deg = 15;
max_term = 10;
var_lb = -1e3;
var_ub = 1e3;
tol_x = 1e-10;
options = optimset('TolX', tol_x);
precision_x = tol_x;
benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

num_var = 10;
max_deg = 20;
max_term = 20;
var_lb = -1e2;
var_ub = 1e2;
tol_x = 1e-10;
options = optimset('TolX', tol_x);
precision_x = tol_x;
benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

num_var = 30;
max_deg = 60;
max_term = 30;
var_lb = -1;
var_ub = 1;
tol_x = 1e-10;
options = optimset('TolX', tol_x);
precision_x = tol_x;
benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y);

function tb = benchmark_mp(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, options, precision_x, precision_y)
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
    
    nlp = 'nlp';
    root = 'root';

    time_con = 0;
    time_gsccd_nlp = 0;
    time_gsccd_root = 0;

    con_eq_nlp = 0;
    con_beat_nlp = 0;
    con_lose_nlp = 0;
    
    con_eq_root = 0;
    con_beat_root = 0;
    con_lose_root = 0;

    nlp_eq_root = 0;
    nlp_beat_root = 0;
    nlp_lose_root = 0;

    for i = 1:batch_size
        p = rand_mp_prob(num_var, max_deg, max_term, coeff_ub, coeff_lb, var_lb, var_ub, options);
        f = p.objective;

        p.options.Solver = 'fmincon';
        t = clock;
        [T,x_con] = evalc('double(mp_solve(p))');
        time_con = time_con + etime(clock, t);
        
        p.options.Solver = 'gsccd';
        p.options.UpSolver = 'nlp';
        t = clock;
        [T,x_gsccd_nlp] = evalc('double(mp_solve(p))');
        time_gsccd_nlp = time_gsccd_nlp + etime(clock, t);
        
        p.options.Solver = 'gsccd';
        p.options.UpSolver = 'root';
        t = clock;
        [T,x_gsccd_root] = evalc('double(mp_solve(p))');
        time_gsccd_root = time_gsccd_root + etime(clock, t);

        % compare objective value
        y_con = double(f(x_con));
        y_gsccd_nlp = double(f(x_gsccd_nlp));
        y_gsccd_root = double(f(x_gsccd_root));

        
        if isEq(x_con, x_gsccd_nlp, precision_x, y_con, y_gsccd_nlp, precision_y)
            con_eq_nlp = con_eq_nlp + 1;
        elseif y_con < y_gsccd_nlp
            con_beat_nlp = con_beat_nlp + 1;
        else
            con_lose_nlp = con_lose_nlp + 1;
        end

        if isEq(x_con, x_gsccd_root, precision_x, y_con, y_gsccd_root, precision_y)
            con_eq_root = con_eq_root + 1;
        elseif y_con < y_gsccd_root
            con_beat_root = con_beat_root + 1;
        else
            con_lose_root = con_lose_root + 1;
        end

        if isEq(x_gsccd_nlp, x_gsccd_root, precision_x, y_gsccd_nlp, y_gsccd_root, precision_y)
            nlp_eq_root = nlp_eq_root + 1;
        elseif y_gsccd_nlp < y_gsccd_root
            nlp_beat_root = nlp_beat_root + 1;
        else
            nlp_lose_root = nlp_lose_root + 1;
        end
    end

    fprintf('\n');
    disp(table([time_con], [time_gsccd_nlp], [time_gsccd_root], 'VariableNames', {'fmincon', 'gsccd_nlp', 'gsccd_root'}, 'RowNames', {'time'}));
    fprintf('\n');
    disp(table([con_beat_nlp/batch_size], [con_lose_nlp/batch_size], [con_eq_nlp/batch_size], 'VariableNames', {'fmincon', 'gsccd_nlp', 'equal'}, 'RowNames', {'% win'}));
    fprintf('\n');
    disp(table([con_beat_root/batch_size], [con_lose_root/batch_size], [con_eq_root/batch_size], 'VariableNames', {'fmincon', 'gsccd_root', 'equal'}, 'RowNames', {'% win'}));
    fprintf('\n');
    disp(table([nlp_beat_root/batch_size], [nlp_lose_root/batch_size], [nlp_eq_root/batch_size], 'VariableNames', {'gsccd_nlp', 'gsccd_root', 'equal'}, 'RowNames', {'% win'}));
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
    nlp_eq_root = 0;
    nlp_beat_root = 0;
    nlp_lose_root = 0;
    for i = 1:batch_size
        p = rand_up_prob(max_deg, coeff_lb, coeff_ub, var_lb, var_ub, options);
        f = p.objective;

        p.options.Solver = 'nlp';
        t = clock;
        x_nlp = up_solve(p);
        time_nlp = time_nlp + etime(clock, t);
        
        p.options.Solver = 'root';
        t = clock;
        x_root = up_solve(p);
        time_root = time_root + etime(clock, t);
       
        y_nlp = double(f(x_nlp));
        y_root = double(f(x_root));

        if isEq(x_nlp, x_root, precision_x, y_nlp, y_root, precision_y)
            nlp_eq_root = nlp_eq_root + 1;
        elseif y_root < y_nlp
            nlp_lose_root = nlp_lose_root + 1;
        else
            nlp_beat_root = nlp_beat_root + 1;
        end
    end
    
    fprintf('\n');
    disp(table([time_nlp], [time_root], 'VariableNames', {'fminbnd', 'root'}, 'RowNames', {'time'}));
    fprintf('\n');
    disp(table([nlp_beat_root/batch_size], [nlp_lose_root/batch_size], [nlp_eq_root/batch_size], 'VariableNames', {'fminbnd', 'root', 'equal'}, 'RowNames', {'% win'}));
    fprintf('\n');
end

%% multivariate polynomial solver
function x = mp_solve(p)
    if ~isfield(p.options, 'Solver')
        p.options.Solver = 'fmincon';
    end

    if strcmp(p.options.Solver, 'fmincon')
        x = mp_solve_fmincon(p);
    elseif strcmp(p.options.Solver, 'gsccd')
        x = mp_solve_gsccd(p);
    else
        error('Unrecognized multivariate polynomial solver');
    end
end

%% wrapper of fmincon
function x = mp_solve_fmincon(p)
    p.solver = 'fmincon';
    x = fmincon(p);
end

%% Gauss-Seidel cyclic coordinate descent methods
function x = mp_solve_gsccd(p)
    if ~isfield(p.options, 'UpSolver')
        p.options.UpSolver = 'nlp';
    end

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
            up_options = p.options;
            up_options.Solver = p.options.UpSolver;
            up = struct('objective', up_f, 'x1', p.box(i,1), 'x2', p.box(i,2), 'options', up_options);
            x_opt = up_solve(up);
            x(i) = x_opt;
            if x(i) == x_old(i)
                done.add(i);
            end
        end
    end
end

%% univariate polynomial solver
function x = up_solve(p)
    if ~isfield(p.options, 'Solver')
        p.options.Solver = 'nlp';
    end

    if strcmp(p.options.Solver, 'nlp')
        x = up_solve_nlp(p);
    elseif strcmp(p.options.Solver, 'root')
        x = up_solve_root(p);
    else
        error('Unrecognized univariate polynomial solver');
    end
end

%% univariate polynomial solver by non-linear programming solver
% f: function handle of univariate polynomial
function x = up_solve_nlp(p)
    f = p.objective;
    p.solver = 'fminbnd';
    local_minimizer = fminbnd(p);
    [min_val, min_idx] = min([f(local_minimizer); f(p.x1); f(p.x2)]);
    candidates = [local_minimizer p.x1 p.x2];
    x = candidates(min_idx);
end

%% univariate polynomial solver by finding root of derivative
% f: function handle of univariate polynomial
function x = up_solve_root(p)
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
    x = candidates(min_idx);
end

%% generate a random multivariate polynomial minimization problem
function prob = rand_mp_prob(num_var, max_deg, max_term, coeff_ub, coeff_lb, var_lb, var_ub, options)
    prob.sym_obj = rand_mulpoly(num_var, max_deg, max_term, coeff_ub, coeff_lb);
    prob.objective = mulpoly2fun(prob.sym_obj);
    prob.x0 = rand_initpoint(num_var, var_lb, var_ub);
    prob.box = rand_box(num_var, var_lb, var_ub);
    [A b] = box2constraint(prob.box);
    prob.Aineq = A;
    prob.bineq = b;
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
function x0 = rand_initpoint(num_var, var_lb, var_ub)
    for i = 1:num_var
        x0(i) = (var_ub - var_lb) * rand() + var_lb;
    end
end

%% convert a box into constraint of form Ax <= b
function [A b] = box2constraint(box)
    for i = 1:length(box)
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
function eq = isEq(x_base, x, precision_x, y_base, y, precision_y)
    eq = or(norm(x_base - x) <= precision_x, abs(y_base - y) <= abs(y_base * precision_y));
end