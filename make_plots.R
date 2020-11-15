source('./errorBarsE.R')

data <- read.csv(file = 'results__8.csv')

data <- transform(data, naive = ((average_performance_original_FSM - average_simulation) * (average_performance_original_FSM - average_simulation)) / average_simulation)
data <- transform(data, d_WIS = ((discounted_WIS - average_simulation) * (discounted_WIS - average_simulation)) / average_simulation)
data <- transform(data, d_OIS = ((discounted_OIS - average_simulation) * (discounted_OIS - average_simulation)) / average_simulation)
data <- transform(data, p_d_WIS = ((proportional_discounted_WIS - average_simulation) * (proportional_discounted_WIS - average_simulation)) / average_simulation)
data <- transform(data, p_d_OIS = ((proportional_discounted_OIS - average_simulation) * (proportional_discounted_OIS - average_simulation)) / average_simulation)
data <- transform(data, ud_WIS = ((WIS - average_simulation) * (WIS - average_simulation)) / average_simulation)
data <- transform(data, ud_OIS = ((OIS - average_simulation) * (OIS - average_simulation)) / average_simulation)
data <- transform(data, ud_p_WIS = ((proportional_WIS - average_simulation) * (proportional_WIS - average_simulation)) / average_simulation)
data <- transform(data, ud_p_OIS = ((proportional_OIS - average_simulation) * (proportional_OIS - average_simulation)) / average_simulation)

data

error.barsEnhanced(data[, c("naive", "d_WIS", "p_d_WIS", "ud_WIS", "ud_p_WIS") ], xlab="test", ylim=c(0, 15), comparing.lines=TRUE, yaxt="s")
