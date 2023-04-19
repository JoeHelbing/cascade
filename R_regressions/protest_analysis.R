library(ggplot2)
library(dplyr)
library(tidyverse)
library(lubridate)
library(knitr)
library(stargazer)
library(ggfortify)
library(gridExtra)
library(naniar)




# read data
df <- read.csv("prot_country_year_pop_mil.csv")

# look at NAs
gg_miss_var(df)

# table and image path
table_path <- "tables/"

# only autocracies by v2x_polyarchy less than 0.5
df_poly_aut <- df[df$v2x_polyarchy < 0.5, ]

# subset df_poly_aut to only include countries with less than 120 protests per million people
df_poly_aut_sub <- df_poly_aut[df_poly_aut$per_mil < 120, ]

################################################################################
# PLOTS IN PAPER #
################################################################################

# log transformed plot per_mil by v2x_polyarchy
plot_poly_aut <- ggplot(df_poly_aut, aes(x = v2x_polyarchy, y = log_per_mil)) +
    geom_point() +
    geom_smooth(method = "lm", se = TRUE, color = "red") +
    labs(x = "Polyarchy Score", y = "Log of Protests per Million People") +
    ggtitle("Log of Protests per Million People by V-Dem Polyarchy Score of Authoritarian Countries") +
    theme(
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        plot.title = element_text(size = 20)
    )

# Display the plot
plot_poly_aut

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Polyarchy Score of Authoritarian Countries.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_poly_aut, width = 10, height = 6, dpi = 300)


# linear plot per_mil by v2x_polyarchy
lin_plot_poly_aut <- ggplot(
    df_poly_aut,
    aes(x = v2x_polyarchy, y = per_mil, color = country)
) +
    geom_point() +
    geom_smooth(method = "lm", se = TRUE, color = "red") +
    labs(x = "Polyarchy Score", y = "Protests per Million People") +
    ggtitle("Protests per Million People by V-Dem Polyarchy Score of Authoritarian Countries") +
    theme(
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        plot.title = element_text(size = 20)
    )

# Display the plot
lin_plot_poly_aut

# Save the plot as an image
image_name <- "Protests per Million People by V-Dem Polyarchy Score of Authoritarian Countries.png"
ggsave(filename = paste0(table_path, image_name), plot = lin_plot_poly_aut, width = 10, height = 6, dpi = 300)

# log transformed plot per_mil by v2x_polyarchy where per mil is less than 120
plot_poly_aut_sub <- ggplot(df_poly_aut_sub, aes(x = v2x_polyarchy, y = log_per_mil)) +
    geom_point() +
    geom_smooth(method = "lm", se = TRUE, color = "red") +
    labs(x = "Polyarchy Score", y = "Log of Protests per Million People") +
    ggtitle("Log Transform Protests per Million People (no outliers < 120 per mil)") +
    theme(
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        plot.title = element_text(size = 20)
    )

# Display the plot
plot_poly_aut_sub

# Linear plot per_mil by v2x_polyarchy where per mil is less than 120
lin_plot_poly_aut_sub <- ggplot(df_poly_aut_sub, aes(x = v2x_polyarchy, y = per_mil)) +
    geom_point() +
    geom_smooth(method = "lm", se = TRUE, color = "red") +
    labs(x = "Polyarchy Score", y = "Protests per Million People") +
    ggtitle("Protests per Million People (no outliers < 120 per mil)") +
    theme(
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        plot.title = element_text(size = 20)
    )

# Display the plot
lin_plot_poly_aut_sub

################################################################################
# MODELS IN PAPER #
################################################################################

# expanding v2x_polyarchy by 100
df_poly_aut$v2x_polyarchy_100 <- 100 * df_poly_aut$v2x_polyarchy
# adding a quadratic term for v2x_polyarchy
df_poly_aut$v2x_polyarchy_sq <- df_poly_aut$v2x_polyarchy_100^2
# sqaure all covariates
df_poly_aut$gdp_growth_sq <- df_poly_aut$gdp_growth^2
df_poly_aut$gdp_growth_lag_sq <- df_poly_aut$gdp_growth_lag^2
df_poly_aut$gdppc_growth_sq <- df_poly_aut$gdppc_growth^2
df_poly_aut$gdppc_growth_lag_sq <- df_poly_aut$gdppc_growth_lag^2
df_poly_aut$standardized_gdp_sq <- df_poly_aut$standardized_gdp^2
df_poly_aut$gdppc_sq <- df_poly_aut$gdppc^2

nrow(df_poly_aut) - nrow(df_poly_aut_sub)

# selected covs log transformed
model_poly_log <- lm(
    log_per_mil ~ v2x_polyarchy_100 +
        gdp_growth +
        gdp_growth_lag +
        gdppc_growth +
        gdppc_growth_lag +
        standardized_gdp +
        gdppc,
    data = df_poly_aut[df_poly_aut$per_mil < 120, ]
)
summary(model_poly_log, digits = 3)
stargazer(model_poly_log, type = "text", title = "Model 3: Polyarchy Autocracy with Covariates Log Transformed")

latex_filename <- paste0(table_path, "Model_2_Polyarchy_Autocracy_with_Covariates_Log_Transformed.html")
stargazer(model_poly_log, type = "html", title = "Model 2: Polyarchy Autocracy with Covariates Log Transformed", out = latex_filename)

# ggplot model_poly_log residuals with color by country
diagnostic_plots <- autoplot(model_poly_log, ncol = 2, nrow = 2)
diagnostic_plots

# Extract each plot from the diagnostic_plots object
plot_1 <- diagnostic_plots[[1]]
plot_2 <- diagnostic_plots[[2]]
plot_3 <- diagnostic_plots[[3]]
plot_4 <- diagnostic_plots[[4]]

# Combine the plots in a 2x2 grid
combined_plot <- gridExtra::arrangeGrob(plot_1, plot_2, plot_3, plot_4, ncol = 2, nrow = 2)

# Save the combined plot
image_filename <- paste0(table_path, "Model_2_Polyarchy_Autocracy_with_Covariates_Log_Transformed.png")
ggsave(image_filename, combined_plot, width = 10, height = 10, units = "in")

model_poly_log <- lm(
    log_per_mil ~ v2x_polyarchy_100 +
        v2x_polyarchy_sq +
        gdp_growth +
        gdp_growth_sq +
        gdp_growth_lag +
        gdp_growth_lag_sq +
        gdppc_growth +
        gdppc_growth_sq +
        gdppc_growth_lag +
        gdppc_growth_lag_sq +
        standardized_gdp +
        standardized_gdp_sq +
        gdppc +
        gdppc_sq,
    data = df_poly_aut
)
summary(model_poly_log, digits = 3)
# stargazer console output with std. errors, t values, and p values
stargazer(model_poly_log, type = "text", title = "Model 3: Polyarchy Autocracy with Covariates Log Transformed")
diagnostic_plots <- autoplot(model_poly_log, ncol = 2, nrow = 2)
diagnostic_plots


# No covriates log transformed
model_poly_log <- lm(
    log_per_mil ~ v2x_polyarchy_100,
    data = df_poly_aut[df_poly_aut$per_mil < 120, ]
)
summary(model_poly_log, digits = 5)
latex_filename <- paste0(table_path, "Model_1_Polyarchy_Autocracy_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 1: Polyarchy Autocracy no Covariates Log Transformed", out = latex_filename)
diagnostic_plots <- autoplot(model_poly_log, ncol = 2, nrow = 2)
diagnostic_plots

# Extract each plot from the diagnostic_plots object
plot_1 <- diagnostic_plots[[1]]
plot_2 <- diagnostic_plots[[2]]
plot_3 <- diagnostic_plots[[3]]
plot_4 <- diagnostic_plots[[4]]

# Combine the plots in a 2x2 grid
combined_plot <- gridExtra::arrangeGrob(plot_1, plot_2, plot_3, plot_4, ncol = 2, nrow = 2)

# Save the combined plot
image_filename <- paste0(table_path, "Model_1_Polyarchy_Autocracy_no_Covariates_Log_Transformed.png")
ggsave(image_filename, combined_plot, width = 10, height = 10, units = "in")



################################################################################
# Experiments #
################################################################################
# only autocracies by v2x_libdem less than 0.5
df_libdem_aut <- df[df$v2x_libdem < 0.5, ]

# only autocracies by v2x_partipdem less than 0.5
df_partipdem_aut <- df[df$v2x_partipdem < 0.5, ]

# only autocracies by v2x_delibdem less than 0.5
df_delibdem_aut <- df[df$v2x_delibdem < 0.5, ]

# only autocracies by v2x_egaldem less than 0.5
df_egaldem_aut <- df[df$v2x_egaldem < 0.5, ]


# plot log_per_mil by v2x_polyarchy full dataset
ggplot(df, aes(x = v2x_polyarchy, y = log_per_mil)) +
    geom_point() +
    geom_smooth(method = "lm") +
    labs(x = "Polyarchy Score", y = "Log of Protests per Million People") +
    ggtitle("Log of Protests per Million People by Polyarchy Vdem Score 0 to 1") +
    theme(
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        plot.title = element_text(size = 20)
    )


# linear plot of full dataset polyarchy
ggplot(df, aes(x = v2x_polyarchy, y = per_mil)) +
    geom_point() +
    geom_smooth(method = "lm") +
    labs(x = "Polyarchy Score", y = "Protests per Million People") +
    ggtitle("Protests per Million People by Polyarchy Vdem Score 0 to 1") +
    theme(
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        plot.title = element_text(size = 20)
    )

# plot log_per_mil by v2x_libdem
ggplot(df_libdem_aut, aes(x = v2x_libdem, y = log_per_mil)) +
    geom_point() +
    geom_smooth(method = "lm") +
    labs(x = "Liberal Democracy Score", y = "Log of Protests per Million People") +
    ggtitle("Log of Protests per Million People by Liberal Democracy Vdem Score 0 to 0.5") +
    theme(
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        plot.title = element_text(size = 20)
    )


# plot log_per_mil by v2x_partipdem
ggplot(df_partipdem_aut, aes(x = v2x_partipdem, y = log_per_mil)) +
    geom_point() +
    geom_smooth(method = "lm") +
    labs(x = "Participatory Democracy Score", y = "Log of Protests per Million People") +
    ggtitle("Log of Protests per Million People by Participatory Democracy Vdem Score 0 to 0.5") +
    theme(
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        plot.title = element_text(size = 20)
    )

# plot log_per_mil by v2x_delibdem
ggplot(df_delibdem_aut, aes(x = v2x_delibdem, y = log_per_mil)) +
    geom_point() +
    geom_smooth(method = "lm") +
    labs(x = "Deliberative Democracy Score", y = "Log of Protests per Million People") +
    ggtitle("Log of Protests per Million People by Deliberative Democracy Vdem Score 0 to 0.5") +
    theme(
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        plot.title = element_text(size = 20)
    )

# plot log_per_mil by v2x_egaldem
ggplot(df_egaldem_aut, aes(x = v2x_egaldem, y = log_per_mil)) +
    geom_point() +
    geom_smooth(method = "lm") +
    labs(x = "Equalitarian Democracy Score", y = "Log of Protests per Million People") +
    ggtitle("Log of Protests per Million People by Equalitarian Democracy Vdem Score 0 to 0.5") +
    theme(
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        plot.title = element_text(size = 20)
    )


# adding a quadratic term for v2x_polyarchy
df_poly_aut$v2x_polyarchy_sq <- df_poly_aut$v2x_polyarchy^2
# add a quadratic term gdp_growth_lag
df_poly_aut$gdp_growth_lag_sq <- df_poly_aut$gdp_growth_lag^2

# linear non-logged
model_poly <- lm(
    per_mil ~ v2x_polyarchy_100,
    data = df_poly_aut
)
summary(model_poly, digits = 5)
latex_filename <- paste0(table_path, "Model_3_Polyarchy_Autocracy_no_Covariates.tex")
stargazer(model_poly, type = "latex", title = "Model 3: Polyarchy Autocracy no Covariates", out = latex_filename)
diagnostic_plots <- autoplot(model_poly, ncol = 2, nrow = 2)

# Extract each plot from the diagnostic_plots object
plot_1 <- diagnostic_plots[[1]]
plot_2 <- diagnostic_plots[[2]]
plot_3 <- diagnostic_plots[[3]]
plot_4 <- diagnostic_plots[[4]]

# Combine the plots in a 2x2 grid
combined_plot <- gridExtra::arrangeGrob(plot_1, plot_2, plot_3, plot_4, ncol = 2, nrow = 2)

# Save the combined plot
image_filename <- paste0(table_path, "Model_3_Polyarchy_Autocracy_no_Covariates.png")
ggsave(image_filename, combined_plot, width = 10, height = 10, units = "in")

# linear non-logged with covariates
model_poly <- lm(
    per_mil ~ v2x_polyarchy_100 +
        gdp_growth +
        gdp_growth_lag +
        gdppc_growth +
        gdppc_growth_lag +
        standardized_gdp +
        gdppc,
    data = df_poly_aut
)
summary(model_poly, digits = 5)
latex_filename <- paste0(table_path, "Model_4_Polyarchy_Autocracy_with_Covariates.tex")
stargazer(model_poly, type = "latex", title = "Model 4: Polyarchy Autocracy with Covariates", out = latex_filename)
diagnostic_plots <- autoplot(model_poly, ncol = 2, nrow = 2)

# Extract each plot from the diagnostic_plots object
plot_1 <- diagnostic_plots[[1]]
plot_2 <- diagnostic_plots[[2]]
plot_3 <- diagnostic_plots[[3]]
plot_4 <- diagnostic_plots[[4]]

# Combine the plots in a 2x2 grid
combined_plot <- gridExtra::arrangeGrob(plot_1, plot_2, plot_3, plot_4, ncol = 2, nrow = 2)

# Save the combined plot
image_filename <- paste0(table_path, "Model_4_Polyarchy_Autocracy_with_Covariates.png")
ggsave(image_filename, combined_plot, width = 10, height = 10, units = "in")


# lagged cov
model_poly_log <- lm(
    log_per_mil ~ v2x_polyarchy +
        inflation_lag +
        gdp_growth_lag +
        gdp_growth_lag_sq +
        gdppc_growth_lag,
    data = df_poly_aut
)
# create stargazer table of model with standard error, t-stat, p value
stargazer(model_poly_log, type = "text", title = "Model 1: Polyarchy Autocracy")
plot(model_poly_log)

# unlagged covs
model_poly_log <- lm(
    log_per_mil ~ v2x_polyarchy +
        inflation +
        gdp_growth +
        gdppc_growth,
    data = df_poly_aut
)
stargazer(model_poly_log, type = "text", title = "Model 1: Polyarchy Autocracy")
plot(model_poly_log)

# adding quadratic term to model
model_poly_log <- lm(
    log_per_mil ~ v2x_polyarchy +
        v2x_polyarchy_sq +
        inflation +
        inflation_lag +
        gdp_growth +
        gdp_growth_lag +
        gdppc_growth +
        gdppc_growth_lag +
        gdp +
        gdppc,
    data = df_poly_aut
)
stargazer(model_poly_log, type = "text", title = "Model 2: Polyarchy Autocracy")
plot(model_poly_log)
# simple linear models
model_poly_log <- lm(
    log_per_mil ~ v2x_polyarchy +
        inflation +
        inflation_lag +
        gdp_growth +
        gdp_growth_lag +
        gdppc_growth +
        gdppc_growth_lag +
        gdp +
        gdppc,
    data = df_poly_aut
)
summary(model_poly_log)

# plot model_poly
plot(model_poly_log)


model_libdem <- lm(log_per_mil ~ v2x_libdem, data = df_libdem_aut)
summary(model_libdem)

model_partipdem <- lm(log_per_mil ~ v2x_partipdem, data = df_partipdem_aut)
summary(model_partipdem)

model_delibdem <- lm(log_per_mil ~ v2x_delibdem, data = df_delibdem_aut)
summary(model_delibdem)

model_egaldem <- lm(log_per_mil ~ v2x_egaldem, data = df_egaldem_aut)
summary(model_egaldem)
