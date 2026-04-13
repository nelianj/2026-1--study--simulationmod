using DrWatson
@quickactivate "project"
using DataFrames, CSV, Plots
gr(fmt=:png)

df_classic = CSV.read(datadir("dining_classic(N=2).csv"), DataFrame)
df_arbiter = CSV.read(datadir("dining_arbiter(N=2).csv"), DataFrame)
N = 2

df_classic1 = CSV.read(datadir("dining_classic(N=10).csv"), DataFrame)
df_arbiter1 = CSV.read(datadir("dining_arbiter(N=10).csv"), DataFrame)
N1 = 10

eat_cols = [Symbol("Eat_$i") for i = 1:N]
eat_cols1 = [Symbol("Eat_$i") for i = 1:N1]

p1 = plot(
    df_classic.time,
    Matrix(df_classic[:, eat_cols]),
    label = ["Ф $i" for i = 1:N],
    xlabel = "Время",
    ylabel = "Ест (1/0)",
    title = "Классическая сеть",
    )

p2 = plot(
    df_arbiter.time,
    Matrix(df_arbiter[:, eat_cols]),
    label = ["Ф $i" for i = 1:N],
    xlabel = "Время",
    ylabel = "Ест (1/0)",
    title = "Сеть с арбитром",
    )

p3 = plot(
    df_classic.time,
    Matrix(df_classic[:, eat_cols]),
    label = ["Ф $i" for i = 1:N1],
    xlabel = "Время",
    ylabel = "Ест (1/0)",
    title = "Классическая сеть",
    )

p4 = plot(
    df_arbiter.time,
    Matrix(df_arbiter[:, eat_cols]),
    label = ["Ф $i" for i = 1:N1],
    xlabel = "Время",
    ylabel = "Ест (1/0)",
    title = "Сеть с арбитром",
    )

p_final = plot(p1, p2, layout = (2, 1), size = (800, 600))
savefig(plotsdir("final_report(N=2).png"))

p_final1 = plot(p3, p4, layout = (2, 1), size = (800, 600))
savefig(plotsdir("final_report(N=10).png"))
println("Отчёт сохранён в plots/final_report.png")
