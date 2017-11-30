clear; clc;

num_poly = 10;
max_deg = 3;
min_coeff = -100;
max_coeff = 100;
a = -1000;
b = 1000;

% construct random polynomials, as both symbolic function and matlab
% function
polys_sym = {};
polys_func = {};
for i = 1:num_poly
    coeff = randi([min_coeff, max_coeff], 1, max_deg + 1);
    p = poly2sym(coeff);
    polys_sym{length(polys_sym) + 1} = p;
    polys_func{length(polys_func) + 1} = matlabFunction(p);
end

% construct a list of random polynomials
% polys = {};
% for j = 1:num_poly
%     p = @(x) (0);
%     for i = 0:max_deg
%         coeff = randi([min_coeff, max_coeff]);
%         p = @(x) (p(x) + coeff * x^i);
%     end
%     polys{length(polys) + 1} = p;
% end


% non-linear programming solver
time_nlp = 0; 
tic
for i = 1:length(polys_sym)
    local_minimizer = fminbnd(polys_func{i}, a, b);
    [nlp_min_val, nlp_min_idx] = min([polys_func{i}(local_minimizer); polys_func{i}(a); polys_func{i}(b)]);
    nlp_min_val;
    candidates = [local_minimizer a b];
    global_minimizer = candidates(nlp_min_idx);
%     time_nlp = time_nlp + timeit(@() fminbnd(polys{i}, a, b));
%     time_nlp = time_nlp + timeit(@() min([polys{i}(local_minimizer); polys{i}(a); polys{i}(b)]));
end
toc

% find root of derivatives
syms x
tic
for i = 1:length(polys_sym)
    df = diff(polys_sym{i}, x);
    all_roots = roots(sym2poly(df));
    real_roots = all_roots(imag(all_roots) == 0)';
    candidates = horzcat(real_roots, a, b);
    root_min_idx = 1;
    root_min_val = polys_func{i}(candidates(1));
    for j = 2:length(candidates)
        val = polys_func{i}(candidates(j));
        if (val < root_min_val)
            root_min_val = val;
            root_min_idx = j;
        end
    end
    root_min_val;
    minimizer = candidates(root_min_idx);
end
toc

% SOS
tic
for i=1:length(polys_sym)
    syms x gam;
    vartable = [x];
    prog = sosprogram(vartable);
    prog = sosdecvar(prog,[gam]);
    f = poly2sym(polys_sym(i), x);
    prog = sosineq(prog,(f-gam), [a,b]);
    prog = sossetobj(prog,-gam);
    evalc('prog = sossolve(prog)');
    evalc('SOLgamma = sosgetsol(prog,gam)');
end
toc