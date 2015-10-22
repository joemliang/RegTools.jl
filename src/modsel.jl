using GLM, DataFrames
include("misc.jl")

abstract FitGoodness
abstract InfoMeasure <: FitGoodness

type AIC <: InfoMeasure
end
type AICc <: InfoMeasure
end
type BIC <: InfoMeasure
end

function AIC(dfrm::RegressionModel)
    n = size(dfrm.mf.df)[1] # number of cases
    if dfrm.mf.terms.intercept
        k = length(dfrm.mf.terms.terms) + 1
    else
        k = length(dfrm.mf.terms.terms)
    end
    # number of parameters, including intercept
    return n*log(deviance(dfrm.model.rr)/n) + 2*k
end

function AICc(dfrm::RegressionModel)
    n = size(dfrm.mf.df)[1]
    if dfrm.mf.terms.intercept
        k = length(dfrm.mf.terms.terms) + 1
    else
        k = length(dfrm.mf.terms.terms)
    end
    return AIC(dfrm) + (2*k*(k+1)) / (n-k-1)
end

function BIC(dfrm::RegressionModel)
    n = size(dfrm.mf.df)[1]
    if dfrm.mf.terms.intercept
        k = length(dfrm.mf.terms.terms) + 1
    else
        k = length(dfrm.mf.terms.terms)
    end
    return n*log(deviance(dfrm.model.rr)/n) + log(n)*k
end

type extractrhs
    rhsarray::Array
    rhsstring::AbstractString
end


# Extract the right hand side of a RegressionModel formula
function extractrhs(dfrm::RegressionModel)
    regressors = dfrm.mf.terms.terms
    rhsarray = similar(regressors, AbstractString)
    for i in 1:length(regressors)
        rhsarray[i] = string(regressors[i])
        rhsarray[i] = replace(rhsarray[i], " ", "")
    end
    rhsstring = ""
    for var in rhsarray
        rhsstring = rhsstring*"+"*var
    end
    if !dfrm.mf.terms.intercept
        rhsarray = [0, rhsarray]
        rhsstring = "0+"*rhsstring
    end
    return extractrhs(rhsarray, rhsstring)
end





abstract VarSel

type add1 <: VarSel
    aic::Float64
    add::AbstractString
    model::RegressionModel
end

type drop1 <: VarSel
    aic::Float64
    drop::AbstractString
    model::RegressionModel
end

function add1(dfrm::RegressionModel, scope::AbstractString, data::DataFrame)
    # e.g. scope = "X1+X2+X1&X2"
    replace(scope, " ", "") # remove whitespace in scope
    scope = split(scope, "+")
    index = find(notnull, scope)
    scope = scope[index]

    lhs = dfrm.mf.terms.eterms[1] # response
    rhs = extractrhs(dfrm)

    add = ""
    aic = AIC(dfrm)
    model = dfrm
    for var in scope
        if var in rhs.rhsarray
            continue # if var already in model, try next var
        end
        rhsnew = rhs.rhsstring * "+" * var
        rhsnew = rhsnew[2:end]
        rhsnew = parse(rhsnew)
        fnew = Formula(lhs, rhsnew)
        newfit = fit(LinearModel, fnew, data)
        newaic = AIC(newfit)
        if newaic < aic
            aic = newaic
            add = var
            f = fnew
            model = newfit
        end
    end
    if add == ""
        println("No term added")
        return add1(aic, "N/A", dfrm)
    else
        println("Add $add with AIC = $aic")
        return add1(aic, add, model)
    end
end


function drop1(dfrm::RegressionModel, scope::AbstractString)
    # e.g. scope = "X1+X2+X1&X2"
    replace(scope, " ", "") # remove whitespace in scope
    scope = split(scope, "+")
    index = find(notnull, scope)
    scope = scope[index]

    lhs = dfrm.mf.terms.eterms[1] # response
    rhs = extractrhs(dfrm) # extract rhs

    drop = ""
    aic = AIC(dfrm)
    model = dfrm
    for var in scope
        if var in rhs.rhsarray
            var = "+"*var
            rhsnew = replace(rhs.rhsstring, var, "")
        else
            error("$var is not in the original model, please modify scope")
        end
        rhsnew = rhsnew[2:end]
        rhsnew = parse(rhsnew)
        fnew = Formula(lhs, rhsnew)
        newfit = fit(LinearModel, fnew, dfrm.mf.df)
        newaic = AIC(newfit)
        if newaic < aic
            aic = newaic
            drop = var[2:end]
            f = fnew
            model = newfit
        end
    end
    if drop == ""
        println("No term dropped")
        return drop1(aic, "N/A", dfrm)
    else
        println("Drop $drop with AIC = $aic")
    return drop1(aic, drop, model)
    end
end

drop1(dfrm::RegressionModel) = drop1(dfrm, extractrhs(dfrm).rhsstring)


type step <: VarSel
    aic::Float64
    model::RegressionModel
end

function step(dfrm::RegressionModel, scope::AbstractString, data::DataFrame,
              direction::AbstractString, trace::Bool=false)
    directions = ["backward", "forward", "both"]
    if ~ (direction in directions)
        error("direction must be one of 'backward', 'forward', or 'both'")
    end
    # turn scope to a string array
    scope = split(scope, "+")
    index = find(notnull, scope)
    scope = scope[index]
    # initialize variables
    todrop = ""
    aic = AIC(dfrm)
    model = dfrm
    if direction == "backward"
        drop = drop1(dfrm, scope)
        while drop.drop != ""
            scope = replace(scope, drop.drop, "")
            drop = drop1(drop.model, scope)
        end
        if drop == ""
            println("No term dropped")
            return step(drop.aic, dfrm)
        else
            return step(drop.aic, model)
        end

    elseif direction == "forward"
        add = add1(dfrm, scope, data)
        while add.add != ""
            scope = replace(scope, add.add, "")
            add = add1(add.model, scope, data)
        end
        if add == ""
            println("No term dropped")
            return step(add.aic, dfrm)
        else
            return step(add.aic, model)
        end

    else # both
        drop = drop1(dfrm, scope)
        add = add1(dfrm, scope)
        while drop.drop != "" | add.add != ""
            if drop.aic < add.aic
                model = drop.model
                aic = drop.aic
                scopenew = replace(scope, drop.drop, "")
            else
                model = add.model
                aic = add.aic
                scopenew = replace(scope, add.add, "")
            end
            drop = drop1(model, scopenew)
            add = add1(model, scopenew)
        end
        if scopenew == scope
            println("No term changed")
            return step(AIC(dfrm), dfrm)
        else
            return step(aic, model)
        end
    end
end

function step(dfrm::RegressionModel, direction::AbstractString; trace::Bool=false)
    directions = ["backward", "both"]
    if ~ (direction in directions)
        error("If no scope specified, direction must be 'backward' or 'both'")
    end
    todrop = ""
    aic = AIC(dfrm)
    model = dfrm
    if direction == "backward"
        drop = drop1(dfrm)
        while drop.aic < aic
            aic = drop.aic
            todrop = drop.drop
            model = drop.model
            drop = drop1(model)
        end
        if todrop == ""
            println("No term dropped")
            return step(aic, dfrm)
        else
            return step(aic, model)
        end

    else # both
        scope = extractrhs(dfrm).rhsstring
        data = dfrm.mf.df
        toadd = ""
        drop = drop1(dfrm)
        add = add1(dfrm, scope, data)
        while drop.aic < aic || add.aic < aic
            if drop.aic < add.aic
                aic = drop.aic
                todrop = drop.drop
                model = drop.model
            else
                aic = add.aic
                toadd = add.add
                model = add.model
            end
            drop = drop1(model)
            add = add1(model, scope, data)
        end
        if todrop == "" && toadd == ""
            println("No term changed")
            return step(AIC(dfrm), dfrm)
        else
            return step(aic, model)
        end
    end
end



