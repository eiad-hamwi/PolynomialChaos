using Plots
Plots.default(show = false)

"""
    plot_tracking_metrics(metrics; outdir)

Write PNG artifacts for beam evolution: sigma_x, sigma_y, beta_x, beta_y, eta_x,
and a summary panel. `metrics` must have fields: s, sigma_x, sigma_y, beta_x, beta_y, eta_x (each a vector).
"""
function plot_tracking_metrics(metrics; outdir::AbstractString = "test/artifacts")
    s = metrics.s
    sigma_x = metrics.sigma_x
    sigma_y = metrics.sigma_y
    beta_x = metrics.beta_x
    beta_y = metrics.beta_y
    eta_x = metrics.eta_x

    isdir(outdir) || mkpath(outdir)

    function save(p, name)
        file = joinpath(outdir, name)
        Plots.savefig(p, file)
        return file
    end

    # Filter out NaN for plotting (Plots can skip them)
    valid = .!(isnan.(beta_x) .| isnan.(beta_y) .| isnan.(eta_x))
    any(valid) || (valid = fill(true, length(s)))

    files = String[]

    p = Plots.plot(s, sigma_x; xlabel = "s (m)", ylabel = "σₓ (m)", title = "Horizontal beam size", legend = false)
    push!(files, save(p, "sigma_x_evolution.png"))

    p = Plots.plot(s, sigma_y; xlabel = "s (m)", ylabel = "σy (m)", title = "Vertical beam size", legend = false)
    push!(files, save(p, "sigma_y_evolution.png"))

    p = Plots.plot(s, beta_x; xlabel = "s (m)", ylabel = "βₓ (m)", title = "Horizontal beta function", legend = false)
    push!(files, save(p, "beta_x_evolution.png"))

    p = Plots.plot(s, beta_y; xlabel = "s (m)", ylabel = "βy (m)", title = "Vertical beta function", legend = false)
    push!(files, save(p, "beta_y_evolution.png"))

    p = Plots.plot(s, eta_x; xlabel = "s (m)", ylabel = "ηₓ (m)", title = "Horizontal dispersion", legend = false)
    push!(files, save(p, "eta_x_evolution.png"))

    p = Plots.plot(
        Plots.plot(s, sigma_x; xlabel = "s (m)", ylabel = "σₓ", title = "σₓ", legend = false),
        Plots.plot(s, sigma_y; xlabel = "s (m)", ylabel = "σy", title = "σy", legend = false),
        Plots.plot(s, beta_x; xlabel = "s (m)", ylabel = "βₓ", title = "βₓ", legend = false),
        Plots.plot(s, beta_y; xlabel = "s (m)", ylabel = "βy", title = "βy", legend = false),
        Plots.plot(s, eta_x; xlabel = "s (m)", ylabel = "ηₓ", title = "ηₓ", legend = false);
        layout = (5, 1),
        size = (600, 800),
    )
    push!(files, save(p, "summary_evolution.png"))

    return files
end
