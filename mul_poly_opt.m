clear; clc;
format short


batch_size = 10;
num_var = 2;
max_deg = 10;
max_term = 2;
coeff_lb = -1e-2;
coeff_ub = 1e2;
var_lb = -1e3;
var_ub = 1e3;
tol_x = 1e-6;
options = optimset('TolX', 1e-6);
th_x = tol_x;
th_y = 1e-3;
tb1 = benchmark(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, tol_x, options, th_x, th_y);
fprintf('\n');
disp(tb1);

batch_size = 10;
num_var = 3;
max_deg = 10;
max_term = 5;
coeff_lb = -1e4;
coeff_ub = 1e4;
var_lb = -1e4;
var_ub = 1e4;
tol_x = 1e-6;
options = optimset('TolX', 1e-6);
th_x = tol_x;
th_y = 1e-3;
tb2 = benchmark(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, tol_x, options, th_x, th_y);
fprintf('\n');
disp(tb2);

batch_size = 10;
num_var = 5;
max_deg = 15;
max_term = 10;
coeff_lb = -1e6;
coeff_ub = 1e6;
var_lb = -1e3;
var_ub = 1e3;
tol_x = 1e-10;
options = optimset('TolX', tol_x);
th_x = tol_x;
th_y = 1e-3;
tb3 = benchmark(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, tol_x, options, th_x, th_y);
fprintf('\n');
disp(tb3);

batch_size = 10;
num_var = 10;
max_deg = 20;
max_term = 10;
coeff_lb = -1e6;
coeff_ub = 1e6;
var_lb = -1e2;
var_ub = 1e2;
tol_x = 1e-10;
options = optimset('TolX', tol_x);
th_x = tol_x;
th_y = 1e-3;
tb4 = benchmark(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, tol_x, options, th_x, th_y);
fprintf('\n');
disp(tb4);

function tb = benchmark(batch_size, num_var, max_deg, max_term, coeff_lb, coeff_ub, var_lb, var_ub, tol_x, options, th_x, th_y)
    fprintf('\n---------------\n');
    disp(['batch size = ', num2str(batch_size)]);
    disp(['number of variables = ', num2str(num_var)]);
    disp(['max degree = ', num2str(max_deg)]);
    disp(['number of terms ', num2str(max_term)]);
    disp(['interval of coefficient = [', num2str(coeff_lb), ', ', num2str(coeff_ub), ']']);
    disp(['interval of variable = [', num2str(var_lb), ', ', num2str(var_ub), ']']);
    disp(['termination tolerance on x = ', num2str(tol_x)]);
    disp(['precision for comparing x = ', num2str(th_x)]);
    disp(['precision for comparing f(x) = ', num2str(th_y * 100), '%']);
    
    syms a b c d e f g h l m n
    var = [a b c d e f g h l m n];
    
    nlp = 'nlp';
    root = 'root';

    time_con = 0;
    time_gsccd_nlp = 0;
    time_gsccd_root = 0;

    eq_gsccd_nlp = 0;
    lose_gsccd_nlp = 0;
    win_gsccd_nlp = 0;
    avg_lose_gsccd_nlp = 0;
    avg_win_gsccd_nlp = 0;
    avg_diff_gsccd_nlp = 0;

    eq_gsccd_root = 0;
    lose_gsccd_root = 0;
    win_gsccd_root = 0;

    
    for i=1:batch_size
        % generate a random multivariate polynomial
        p = mp_prob(num_var, max_deg, max_term, coeff_ub, coeff_lb, var_lb, var_ub);
        xs = rand(10, num_var);
        ys = [];
        for i = 1:length(xs)
            ys(i) = p.f(xs(i,:));
        end

        % solve and track computation time
        t = clock;
%         x_con = double(mp_solve_fmincon(p));
        [T,x_con] = evalc('double(mp_solve_fmincon(p))');
        time_con = time_con + etime(clock, t);
        
        t = clock;
%         x_gsccd_nlp = double(mp_solve_gsccd(p, tol_x, nlp, options));
        [T,x_gsccd_nlp] = evalc('double(mp_solve_gsccd(p, tol_x, nlp, options))');
%         T
        time_gsccd_nlp = time_gsccd_nlp + etime(clock, t);
        
        t = clock;
%         x_gsccd_root = double(mp_solve_gsccd(p, tol_x, root, options));
        [T,x_gsccd_root] = evalc('double(mp_solve_gsccd(p, tol_x, root, options))');
%         T
        time_gsccd_root = time_gsccd_root + etime(clock, t);

        % compare objective value
        y_con = double(p.f(x_con));
        y_gsccd_nlp = double(p.f(x_gsccd_nlp));
        y_gsccd_root = double(p.f(x_gsccd_root));
        
        fprintf('\n');
        p.mp.exps
        p.f(var(1:num_var))
        x_con
        x_gsccd_nlp
        x_gsccd_root
        y_con
        y_gsccd_nlp
        y_gsccd_root
        
        if isEq(x_con, x_gsccd_nlp, th_x, y_con, y_gsccd_nlp, th_y)
            eq_gsccd_nlp = eq_gsccd_nlp + 1;
        elseif y_con < y_gsccd_nlp % gsccd lose
            lose_gsccd_nlp = lose_gsccd_nlp + 1;
            avg_lose_gsccd_nlp = avg_lose_gsccd_nlp + abs((y_gsccd_nlp - y_con) / y_con);
            avg_diff_gsccd_nlp = avg_diff_gsccd_nlp + (y_gsccd_nlp - y_con) / y_con;
        else % gsccd win
            win_gsccd_nlp = win_gsccd_nlp + 1;
            avg_win_gsccd_nlp = avg_win_gsccd_nlp + abs((y_con - y_gsccd_nlp) / y_con);
            avg_diff_gsccd_nlp = avg_diff_gsccd_nlp + (y_gsccd_nlp - y_con) / y_con;
        end

        if isEq(x_con, x_gsccd_root, th_x, y_con, y_gsccd_nlp, th_y)
            eq_gsccd_root = eq_gsccd_root + 1;
        elseif y_con < y_gsccd_root
            lose_gsccd_root = lose_gsccd_root + 1;
        else
            win_gsccd_root = win_gsccd_root + 1;
        end
        
        for i = 1:length(xs)
            if ys(i) ~= p.f(xs(i,:))
                fprintf('\n\n\n**************\n!!!!!!SOMETHING WRONG!!!\n\n\n');
            end
        end
    %     diff_gsccd_nlp = (y_gsccd_nlp - y_con) / y_con;
    %     diff_gsccd_root = (y_gsccd_root - y_con) / y_con;    
    end

    avg_lose_gsccd_nlp = avg_lose_gsccd_nlp / batch_size;
    avg_win_gsccd_nlp = avg_win_gsccd_nlp / batch_size;
    avg_diff_gsccd_nlp = avg_diff_gsccd_nlp / batch_size;

    time = [time_con; time_gsccd_nlp; time_gsccd_root];
    eq = [1; eq_gsccd_nlp / batch_size; eq_gsccd_root / batch_size];
    lose = [0; lose_gsccd_nlp / batch_size; lose_gsccd_root];
    win = [0; win_gsccd_nlp / batch_size; win_gsccd_root];
    avg_lose = [0; avg_lose_gsccd_nlp; 0];
    avg_win = [0; avg_win_gsccd_nlp; 0];
    avg_diff = [0; avg_diff_gsccd_nlp; 0];
    tb = table(time, eq, lose, win, avg_lose, avg_win, avg_diff, 'VariableNames', {'time', 'eq', 'lose', 'win', 'avg_lose', 'avg_win', 'avg_diff'}, 'RowNames', {'fmincon', 'gsccd_nlp', 'gsccd_root'});
end

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

%     fmincon(f, x0, ...) % pass explicit upper and lower bound
%     syms x y z
%     disp(f([x y z]));
%     disp(mp.exps);
%     disp(mp.coeffs);
%     if abs(f(x_con) - f(x_gsccd_nlp)) > tol_x || abs(f(x_gsccd_nlp) - f(x_gsccd_root)) > tol_x
%     if abs(f(x_gsccd_nlp) - f(x_gsccd_root)) > 0
%     i
%     abs(f(x_con) - f(x_gsccd_nlp))
%     abs(f(x_gsccd_nlp) - f(x_gsccd_root))
%     disp(x_con);
%     disp(x_gsccd_nlp);
%     disp(x_gsccd_root);
%     disp(double(f(x_con)));
%     disp(double(f(x_gsccd_nlp)));
%     disp(double(f(x_gsccd_root)));
%     end

%% compare result of different optimization methods
function eq = isEq(x_base, x, th_x, y_base, y, th_y)
    eq = or(norm(x_base - x) <= th_x, abs(y_base - y) <= abs(y_base * th_y));
end

%% time a function and suppress all output
function t = time_fun(f)
    [T t] = evalc('timeit(f)');
end

%% generate a multivariate polynomial minimization problem
function prob = mp_prob(num_var, max_deg, max_term, coeff_ub, coeff_lb, var_lb, var_ub)
    prob.mp = rand_mulpoly(num_var, max_deg, max_term, coeff_ub, coeff_lb);
    prob.f = mulpoly2fun(prob.mp);
    prob.x0 = rand_initpoint(num_var, var_lb, var_ub);
    prob.box = rand_box(num_var, var_lb, var_ub);
    [A b] = box2constraint(prob.box);
    prob.A = A;
    prob.b = b;
end

%% wrapper of fmincon
function x = mp_solve_fmincon(p)
    x = fmincon(p.f, p.x0, p.A, p.b);
end

%% Gauss-Seidel cyclic coordinate descent methods
function x = mp_solve_gsccd(p, tol_x, up_solver, options)
    f = p.f;
    box = p.box;
    x0 = p.x0;
    
    if ~exist('up_solver', 'var')
        up_solver = 'nlp'; % nlp as default up_solver
    end
    
    variable = sym('variable');
    num_var = length(x0);
    x = x0;
    while ~exist('x_old', 'var') || norm(x - x_old) > tol_x
        x_old = x;
        for i = 1:num_var
            x = horzcat(x(1 : i-1), variable, x(i+1 : num_var)); 
            up_f = matlabFunction(f(x)); % create function handle of univariate polynomial
%             up_f = @(y) f(subs(x, variable, y));
            if nargin(up_f) == 0 % skip function like @() 0.0
                x(i) = x_old(i);
                continue
            end
            a = box(i,1);
            b = box(i,2);
            x_opt = up_solve(up_f, a, b, tol_x, up_solver, options);
            x(i) = x_opt;
        end
    end
end

%% univariate polynomial solver
function x = up_solve(f, a, b, tol_x, solver, options)
    if ~exist('solver', 'var')
        solver = 'nlp'; % nlp as default solver
    end
    if strcmp(solver, 'nlp')
        x = up_solve_nlp(f, a, b, options);
    else
        x = up_solve_root(f, a, b, options);
    end
end

%% univariate polynomial solver by non-linear programming solver
% f: function handle of univariate polynomial
function x = up_solve_nlp(f, a, b, options)
    local_minimizer = fminbnd(f, a, b, options);
    [min_val, min_idx] = min([f(local_minimizer); f(a); f(b)]);
    candidates = [local_minimizer a b];
    x = candidates(min_idx);
end

%% univariate polynomial solver by finding root of derivative
% f: function handle of univariate polynomial
function x = up_solve_root(f, a, b, options)
    syms y
    sym_f = f(y);
    sym_df = diff(sym_f);
    disp(sym_f);
    disp(sym_df);
    all_roots = roots(sym2poly(sym_df));
    real_roots = all_roots(imag(all_roots) == 0)';
    candidates = horzcat(real_roots, a, b);
    candidates = candidates(and(candidates >= a, candidates <= b));
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