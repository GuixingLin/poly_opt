clear; clc;

num_var = 3;
max_deg = 3;
max_term = 3;
coeff_lower = 0;
coeff_upper = 2;
var_lower = -10;
var_upper = 10;
tol_x = 1e-4;

mp = rand_mulpoly(num_var, max_deg, max_term, coeff_upper, coeff_lower);
f = mulpoly2fun(mp);
x0 = rand_initpoint(num_var, var_lower, var_upper);
box = rand_box(num_var, var_lower, var_upper);
[A b] = box2constraint(box);
fmincon(f, x0, A, b); % interval as constraint
% fmincon(f, x0, ...) % pass explicit upper and lower bound
x_opt = mp_solve_gsccd(f, box, x0, tol_x);
x_opt

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

%% Gauss-Seidel cyclic coordinate descent methods
function x = mp_solve_gsccd(f, box, x0, tol_x)
    variable = sym('variable');
    num_var = length(x0);
    x = x0;
    while ~exist('x_old', 'var') || norm(x - x_old) > tol_x
        x_old = x
        for i = 1:num_var
            x = horzcat(x(1 : i-1), variable, x(i+1 : num_var))
            up_f = @(y) subs(f(x), variable, y); % create function handle of univariate polynomial
            a = box(i,1);
            b = box(i,2);
            x_opt = up_solve(up_f, a, b, tol_x, 'root');
            x(i) = x_opt;
        end
    end 
end

%% univariate polynomial solver
function x = up_solve(f, a, b, tol_x, solver)
    if ~exist('solver', 'var')
        solver = nlp; % nlp as default solver
    end
    if strcmp(solver, 'root')
        x = up_solve_root(f, a, b);
    end
end

%% univariate polynomial solver by finding root of derivative
% f: function handle of univariate polynomial
function x = up_solve_root(f, a, b)
    syms y
    sym_f = f(y);
    sym_df = diff(sym_f);
    all_roots = roots(sym2poly(sym_df));
    real_roots = all_roots(imag(all_roots) == 0)';
    candidates = horzcat(real_roots, a, b);
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
function x0 = rand_initpoint(num_var, var_lower, var_upper)
    for i = 1:num_var
        x0(i) = (var_upper - var_lower) * rand() + var_lower;
    end
end

%% convert box constraint into form Ax <= b
function [A b] = box2constraint(box)
    for i = 1:length(box)
        A(2*i-1, i) = -1;
        b(2*i-1, 1) = -box(i, 1);
        A(2*i, i) = 1;
        b(2*i, 1) = box(i, 2);
    end
end

%% generate a random box represented by an nx2 matrix, where i_th row is the interval [a b] for x_i
function box = rand_box(num_var, var_lower, var_upper)
    for i = 1:num_var
        box(i,1) = randi([var_lower, var_upper]); % use integer interval for now
        box(i,2) = randi([box(i,1), var_upper]);
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

%% generate a multivariate polynomial, which contains
% an exponent matrix where each row represents an exponent
% a vector of coefficients
function mp = rand_mulpoly(num_var, max_deg, max_term, coeff_upper, coeff_lower)
    for i = 1:max_term
        m = rand_monomial(num_var, max_deg, coeff_lower, coeff_upper);
        mp.exps(i,:) = m.exp;
        mp.coeffs(i,1) = m.coeff;
    end
end

%% create function handle of a monomial
function fun = monomial2fun(m)
    fun = @(x) (m.coeff)*prod(x.^(m.exp));
end

%% generate a random monomial, which contains
% an exponent represented by a row vector
% a coefficient
function m = rand_monomial(num_var, max_deg, coeff_lower, coeff_upper)
    for i = 1:num_var
        if max_deg > 0
            e = randi([0, max_deg]);
            m.exp(i) = e;
            max_deg = max_deg - e;
        else
            m.exp(i) = 0;
        end
    end
    m.coeff = randi([coeff_lower, coeff_upper]); % use integer coeff for now
%     mo.coeff = (coeff_upper - coeff_lower) * rand() + coeff_lower;
end

%% permute an array in random order
function b = rand_permute(a)
    order = randperm(length(a));
    for i = 1:length(a)
        b(i) = a(order(i));
    end
end