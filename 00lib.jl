function quantize(m::Matrix{T}) where {T <: Number}
    v = filter(!isnan, vec(m))
    Q = StatsBase.ecdf(v).(m)
    for i in eachindex(m)
        if isnan(m[i])
            Q[i] = NaN
        end
    end
    return Q
end

function readasc(f::String; nd="nodata")
    # TODO make sure this handles nodata correctly, this is not the case atm
    asc_lines = readlines(f)
    ncols_line = first(filter(l -> startswith(l, "ncols"), asc_lines))
    ncols = parse(Int64, last(split(ncols_line, " ")))
    nrows_line = first(filter(l -> startswith(l, "nrows"), asc_lines))
    nrows = parse(Int64, last(split(nrows_line, " ")))
    m = zeros(Float64, (ncols, nrows))
    data_start = findfirst(l -> startswith(l, nd), asc_lines)+1
    data_end = length(asc_lines)
    for line_id in data_start:data_end
        d = parse.(Float64, split(asc_lines[line_id]))
        m[:,nrows-(line_id-(data_start))] = d
    end
    current = convert(Matrix, m')
    for i in eachindex(current)
        if current[i] in [0.0, -9999]
            current[i] = NaN
        end
    end
    return quantize(current)
end
