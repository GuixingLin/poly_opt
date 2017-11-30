clear; clc;

num_poly = 1;
max_deg = 4;
min_coeff = -100;
max_coeff = 100;
a = -100;
b = 100;

% construct random polynomials, as both symbolic function and matlab
% function
polys_sym = {};
polys_func = {};
for i = 1:num_poly
    coeff = randi([min_coeff, max_coeff], 1, max_deg + 1);
    p = poly2sym(coeff)
    polys_sym{length(polys_sym) + 1} = p;
    polys_func{length(polys_func) + 1} = matlabFunction(p);
end

% non-linear programming solver
tic
for i = 1:length(polys_sym)
    local_minimizer = fminbnd(polys_func{i}, a, b);
    [min_val_nlp, min_idx_nlp] = min([polys_func{i}(local_minimizer); polys_func{i}(a); polys_func{i}(b)]);
    min_val_nlp;
    candidates = [local_minimizer a b];
%     min_idx_nlp = 1;
%     min_val_nlp = polys_func{i}(candidates(1));
%     for j = 2:length(candidates)
%         val = polys_func{i}(candidates(j));
%         if (val < min_val_nlp)
%             min_val_nlp = val;
%             min_idx_nlp = j;
%         end
%     end
    minimizer_nlp = candidates(min_idx_nlp);
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
    min_idx_root = 1;
    min_val_root = polys_func{i}(candidates(1));
    for j = 2:length(candidates)
        val = polys_func{i}(candidates(j));
        if (val < min_val_root)
            min_val_root = val;
            min_idx_root = j;
        end
    end
    minimizer_root = candidates(min_idx_root);
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
    prog = sosineq(prog,(f-gam), [a b]);
    prog = sossetobj(prog,-gam);
    prog = sossolve(prog);
    SOLgamma = sosgetsol(prog,gam);
end
toc

disp(min_val_nlp)
disp(min_val_root)
disp(double(SOLgamma))