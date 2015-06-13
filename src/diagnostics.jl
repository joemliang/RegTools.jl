using GLM, DataFrames, Gadfly, Distributions
include("misc.jl")

function avPlot(dfrm::RegressionModel, variable::Symbol)
    varNames = names(dfrm.mf.df)
    # get e_yx
    lhs = dfrm.mf.terms.eterms[1]
    rhs = extractrhs(dfrm)
    rhsnew = replace(rhs.rhsstring, "+"*string(variable), "")
    rhsnew = rhsnew[2:end]
    rhsnew = parse(rhsnew)
    fnew = Formula(lhs, rhsnew)
    newfit = fit(LinearModel, fnew, dfrm.mf.df)
    e_yx = residuals(newfit)
    # get e_zx
    X = dfrm.mf.df[2:end]
    Z = X[variable]
    X = X[~[(name in [variable]) for name in names(X)]]
    intercept = ones(Int, nrow(X), 1)
    X = convert(Array, X)
    X = hcat(intercept, X)
    H = *(*(X, inv(*(transpose(X), X))), transpose(X))
    e_zx = *((eye(size(H)[1])-H), Z)
    df = DataFrame(e_yx=e_yx, e_zx=e_zx)
    # get alpha
    avlm = fit(LinearModel, e_yx~0+e_zx, df)
    alpha = coef(avlm)
    # plot
    xs_range = abs(maximum(df[:e_zx]) - minimum(df[:e_zx]))
    xs = linspace(minimum(df[:e_zx])-xs_range/10, maximum(df[:e_zx])+xs_range/10)
    ys = alpha .* xs
    xy = DataFrame(xs=xs, ys=ys)
    plot(layer(df, x="e_zx", y="e_yx", Geom.point), Guide.xlabel(string(variable)*" | others"),
         Guide.ylabel(string(varNames[1])*" | others"),
         Guide.title("Added-Variable Plot"),
         layer(xy, x="xs", y="ys", Theme(default_color=color("red")), Geom.line))
end


function vif(dfrm::RegressionModel)
    X = dfrm.mf.df[2:end]
    rhs = extractrhs(dfrm)
    result = DataFrame(variable=rhs.rhsarray, vif=0.0)
    i = 1
    for (var in rhs.rhsarray)
        lhs = parse(var)
        rhsnew = replace(rhs.rhsstring, "+"*var, "")
        rhsnew = rhsnew[2:end]
        rhsnew = parse(rhsnew)
        fnew = Formula(lhs, rhsnew)
        newfit = fit(LinearModel, fnew, X)
        r2 = rsquared(newfit)
        result[:vif][i] = 1 / (1-r2)
        i = i + 1
    end
    result
end


function rsquared(dfrm::RegressionModel)
    SStot = sum((dfrm.model.rr.y - mean(dfrm.model.rr.y)).^2)
    SSres = sum((dfrm.model.rr.y - dfrm.model.rr.mu).^2)
    return (1-(SSres/SStot))
end


function adjrsquared(dfrm::RegressionModel)
    SStot = sum((dfrm.model.rr.y - mean(dfrm.model.rr.y)).^2)
    SSres = sum((dfrm.model.rr.y - dfrm.model.rr.mu).^2)
    n = size(dfrm.model.rr.y, 1)  #number of samples
    if dfrm.mf.terms.intercept
        p = size(dfrm.mm.m, 2) - 1
    else
        p = size(dfrm.mm.m, 2)
    end
    return 1- ( (SSres/(n-p-1)) / (SStot/(n-1)) )
end



function rstudent(dfrm::RegressionModel)
    SSres = sum((dfrm.model.rr.y - dfrm.model.rr.mu).^2)
    n = size(dfrm.model.rr.y, 1)  #number of samples
    if dfrm.mf.terms.intercept
        p = size(dfrm.mm.m, 2) - 1
    else
        p = size(dfrm.mm.m, 2)
    end
    sigma2 = SSres / (n-p)
    X = dfrm.mm.m
    H = *(*(X, inv(*(transpose(X), X))), transpose(X))
    h = diag(H)
    r = residuals(dfrm) ./ (sqrt(sigma2 .* (1 - h)))
    return r
end

function jackknife(dfrm::RegressionModel)
    r = rstudent(dfrm)
    n = size(dfrm.model.rr.y, 1)  #number of samples
    if dfrm.mf.terms.intercept
        p = size(dfrm.mm.m, 2) - 1
    else
        p = size(dfrm.mm.m, 2)
    end
    t = r .* sqrt((n-p-1) ./ (n-p-r.^2))
    return t
end


function halfnorm(dfrm::RegressionModel)
    N = size(dfrm.model.rr.y, 1)
    n = 1:N
    X = dfrm.mm.m
    H = *(*(X, inv(*(transpose(X), X))), transpose(X))
    h = diag(H)
    labels = sortperm(h)
    h = h[labels]
    U = (N+n) / (2*N+1)
    d = Normal()
    u = quantile(d, U)
    # prepare labels for potential outliers
    df = DataFrame(u=u, h=h, label="")
    outlierlabel = labels[u.>2]
    no_outliers = length(outlierlabel)
    for i in (length(labels)-no_outliers+1):length(labels)
        df[:label][i] = string(labels[i])
    end
    plot(df, x="u", y="h", label="label", Geom.point, Geom.label, Guide.xlabel("Half-normal quantiles"), Guide.ylabel("Leverages"))
end

function cooksdistance(dfrm::RegressionModel; plotit::Bool=false)
    r = rstudent(dfrm)
    X = dfrm.mm.m
    H = *(*(X, inv(*(transpose(X), X))), transpose(X))
    h = diag(H)
    p = size(dfrm.mm.m, 2)
    d = (r.^2 .* h) ./ (p*(1-h))
    if !plotit
        return d
    else
        # identify outliers
        n = length(d)
        cutoff = 4/(n-p)
        labels = 1:n
        outlierlabel = labels[d.>cutoff]
        df = DataFrame(n=labels, d=d, label="")
        for i in outlierlabel
            df[:label][i] = string(i)
        end
        plot(df, x="n", y="d", label="label", Guide.xlabel("Case number"), Guide.ylabel("Cook's distance"), Geom.bar, Geom.label)
    end
end


