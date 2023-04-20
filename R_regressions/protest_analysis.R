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

# drop 'X' column
df <- subset(df, select = -X)

# look at NAs
gg_miss_var(df)

# table and image path
table_path <- "tables/"

names(df)

################################################################################
################################################################################
# PLOTS IN PAPER AND MODELS IN PAPER#
################################################################################
################################################################################

################################################################################
# only autocracies by v2x_polyarchy less than 0.5
df_poly_aut <- df[df$v2x_polyarchy < 0.5, ]

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
    ) +
  theme_bw()

# Display the plot
plot_poly_aut

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Polyarchy Score of Authoritarian Countries.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_poly_aut, width = 10, height = 6, dpi = 300)


# lin reg with econ covar polyarchy < 0.5
# expanding v2x_polyarchy by 100
df_poly_aut$v2x_polyarchy_100 <- 100 * df_poly_aut$v2x_polyarchy

# selected covs log transformed
model_poly_log <- lm(
  log_per_mil ~ v2x_polyarchy_100 +
    gdp_growth +
    gdp_growth_lag +
    gdppc_growth +
    gdppc_growth_lag +
    standardized_gdp +
    gdppc,
  data = df_poly_aut
)
summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_1_Polyarchy_Autocracy_with_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 1: Polyarchy Autocracy with Covariates Log Transformed", out = latex_filename)

  # residuals commented out, but uncomment to check
# # ggplot model_poly_log residuals with color by country
# diagnostic_plots <- autoplot(model_poly_log, ncol = 2, nrow = 2)
# diagnostic_plots
# 
# # Extract each plot from the diagnostic_plots object
# plot_1 <- diagnostic_plots[[1]]
# plot_2 <- diagnostic_plots[[2]]
# plot_3 <- diagnostic_plots[[3]]
# plot_4 <- diagnostic_plots[[4]]
# 
# # Combine the plots in a 2x2 grid
# combined_plot <- gridExtra::arrangeGrob(plot_1, plot_2, plot_3, plot_4, ncol = 2, nrow = 2)
# 
# # Save the combined plot
# image_filename <- paste0(table_path, "Model_2_Polyarchy_Autocracy_with_Covariates_Log_Transformed.png")
# ggsave(image_filename, combined_plot, width = 10, height = 10, units = "in")

################################################################################
# bivariate lin reg of polyarcy < 0.5
# expanding v2x_polyarchy by 100
df$v2x_polyarchy_100 <- 100 * df$v2x_polyarchy

# selected covs log transformed
model_poly_log <- lm(
  log_per_mil ~ v2x_polyarchy_100,
  data = df_poly_aut
)
summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_2_Polyarchy_Autocracy_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 2: Polyarchy Autocracy no Covariates Log Transformed", out = latex_filename)

################################################################################
# pulling NA econ covariate values for df_poly_aut out for analysis seperately
columns_to_check <- c("inflation", "gdp", "gdp_growth", "gdppc", "gdppc_growth", 
                      "standardized_gdp", "gdp_growth_lag", "inflation_lag", 
                      "gdppc_growth_lag" )
df_poly_with_na <- df_poly_aut[apply(df_poly_aut[columns_to_check], 1, function(x) any(is.na(x))), ]

head(df_poly_with_na)
dim(df_poly_with_na)

# log transformed plot per_mil by v2x_polyarchy only econ covar NAs
plot_poly_aut_na <- ggplot(df_poly_with_na, aes(x = v2x_polyarchy, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Polyarchy Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Polyarchy Score of \nAuthoritarian Countries only NA Covar") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

# Display the plot
plot_poly_aut_na

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Polyarchy Score of Authoritarian Countries Using Only Dropped Econ Covariates.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_poly_aut_na, width = 10, height = 6, dpi = 300)


# lin reg with econ covar polyarchy < 0.5
# expanding v2x_polyarchy by 100
df_poly_with_na$v2x_polyarchy_100 <- 100 * df_poly_with_na$v2x_polyarchy

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_polyarchy_100,
  data = df_poly_with_na
)
summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_3_Polyarchy_Autocracy_no_Covariates_Log_Transformed Only Dropped Econ.tex")
stargazer(model_poly_log, type = "latex", title = "Model 3: Polyarchy Autocracy no Covariates Log Transformed Using Only Dropped Econ Covariates", out = latex_filename)

################################################################################
# Full dataset for polyarchy including democracies
# log transformed plot per_mil by v2x_polyarchy only econ covar NAs
plot_polyarchy <- ggplot(df, aes(x = v2x_polyarchy, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Polyarchy Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Polyarchy Score of \nAll Countries Including Democracies") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

# Display the plot
plot_polyarchy

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Polyarchy Score of ALL Countries.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_polyarchy, width = 10, height = 6, dpi = 300)


# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_polyarchy_100,
  data = df
)
summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_4_Polyarchy_All_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 4: Polyarchy All Countries No Covariates Log Transformed", out = latex_filename)



################################################################################
# EXPERIMENTS AND TESTING #
################################################################################

################################################################################
# Full dataset for v2xcl_disc including democracies
# freedom of discussion
# log transformed plot per_mil by v2xcl_disc

plot_free_disc <- ggplot(df, aes(x = v2xcl_disc, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Freedom of Discussion Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Freedom of Discussion Score \nAll Countries Including Democracies") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_free_disc

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Freedom of Discussion Score All Countries Including Democracies.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_free_disc, width = 10, height = 6, dpi = 300)

# multiply by 100
df$v2xcl_disc_100 <- 100 * df$v2xcl_disc

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2xcl_disc_100,
  data = df
)
summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_5_FreeDisc_All_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 5: Free Discussion Score All Countries No Covariates Log Transformed", out = latex_filename)

# only autocracies
df_aut_filtered_free_disc <- df %>% filter(v2xcl_disc < 0.5)

# log transformed plot per_mil by v2xcl_disc
plot_free_disc_aut <- ggplot(df_aut_filtered_free_disc, aes(x = v2xcl_disc, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Freedom of Discussion Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Freedom of Discussion Score \nAuthoritarian Countries Only") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_free_disc_aut

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Freedom of Discussion Score Authoritarian Countries Only.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_free_disc_aut, width = 10, height = 6, dpi = 300)

# multiply by 100
df_aut_filtered_free_disc$v2xcl_disc_100 <- 100 * df_aut_filtered_free_disc$v2xcl_disc

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2xcl_disc_100,
  data = df_aut_filtered_free_disc
)
summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_6_FreeDisc_Aut_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 6: Free Discussion Score Authoritarian No Covariates Log Transformed", out = latex_filename)

################################################################################
# Full dataset for v2x_freexp including democracies
# Freedom of expression index
# log transformed plot per_mil by v2x_freexp

plot_freexp <- ggplot(df, aes(x = v2x_freexp, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Freedom of Expression Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Freedom of Expression Score \nAll Countries Including Democracies") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_freexp

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Freedom of Expression Score All Countries Including Democracies.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_freexp, width = 10, height = 6, dpi = 300)

# multiply by 100
df$v2x_freexp_100 <- 100 * df$v2x_freexp

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_freexp_100,
  data = df
)
summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_7_FreExp_All_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 7: Freedom of Expression Index All Countries No Covariates Log Transformed", out = latex_filename)

# only autocracies
df_aut_filtered_freexp <- df %>% filter(v2x_freexp < 0.5)

# log transformed plot per_mil by v2x_freexp
plot_freexp_aut <- ggplot(df_aut_filtered_freexp, aes(x = v2x_freexp, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Freedom of Expression Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Freedom of Expression Score \nAuthoritarian Countries Only") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_freexp_aut

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Freedom of Expression Score Authoritarian Countries Only.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_freexp_aut, width = 10, height = 6, dpi = 300)

# multiply by 100
df_aut_filtered_freexp$v2x_freexp_100 <- 100 * df_aut_filtered_freexp$v2x_freexp

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_freexp_100,
  data = df_aut_filtered_freexp
)
summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_8_FreExp_Aut_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 8: Freedom of Expression Index Authoritarian Countries No Covariates Log Transformed", out = latex_filename)

################################################################################
# Full dataset for v2x_civlib including democracies
# Civil liberties index 
# log transformed plot per_mil by v2x_civlib

plot_civlib <- ggplot(df, aes(x = v2x_civlib, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Civil Liberties Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Civil Liberties Score \nAll Countries Including Democracies") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_civlib

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Civil Liberties Score All Countries Including Democracies.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_civlib, width = 10, height = 6, dpi = 300)

# multiply by 100
df$v2x_civlib_100 <- 100 * df$v2x_civlib

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_civlib_100,
  data = df
)

summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_9_CivLib_All_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 9: Civil Liberties Index All Countries No Covariates Log Transformed", out = latex_filename)

# only autocracies
df_aut_filtered_civlib <- df %>% filter(v2x_civlib < 0.5)

# log transformed plot per_mil by v2x_civlib
plot_civlib_aut <- ggplot(df_aut_filtered_civlib, aes(x = v2x_civlib, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Civil Liberties Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Civil Liberties Score \nAuthoritarian Countries Only") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_civlib_aut

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Civil Liberties Score Authoritarian Countries Only.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_civlib_aut, width = 10, height = 6, dpi = 300)

# multiply by 100
df_aut_filtered_civlib$v2x_civlib_100 <- 100 * df_aut_filtered_civlib$v2x_civlib

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_civlib_100,
  data = df_aut_filtered_civlib
)

summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_10_CivLib_Aut_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 10: Civil Liberties Score Authoritarian Countries No Covariates Log Transformed", out = latex_filename)

################################################################################
# Full dataset for v2x_clpol including democracies
# Political civil liberties index 
# log transformed plot per_mil by v2x_clpol

plot_clpol <- ggplot(df, aes(x = v2x_clpol, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Political Civil Liberties Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Political Civil Liberties Score \nAll Countries Including Democracies") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_clpol

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Political Civil Liberties Score All Countries Including Democracies.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_clpol, width = 10, height = 6, dpi = 300)

# multiply by 100
df$v2x_clpol_100 <- 100 * df$v2x_clpol

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_clpol_100,
  data = df
)

summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_11_CLPol_All_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 11: Political Civil Liberties Score All Countries No Covariates Log Transformed", out = latex_filename)

# only autocracies
df_aut_filtered_clpol <- df %>% filter(v2x_clpol < 0.5)

# log transformed plot per_mil by v2x_clpol
plot_clpol_aut <- ggplot(df_aut_filtered_clpol, aes(x = v2x_clpol, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Political Civil Liberties Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Political Civil Liberties Score \nAuthoritarian Countries Only") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_clpol_aut

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Political Civil Liberties Score Authoritarian Countries Only.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_clpol_aut, width = 10, height = 6, dpi = 300)

# multiply by 100
df_aut_filtered_clpol$v2x_clpol_100 <- 100 * df_aut_filtered_clpol$v2x_clpol

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_clpol_100,
  data = df_aut_filtered_clpol
)

summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_12_CLPol_Aut_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 12: Political Civil Liberties Score Authoritarian Countries No Covariates Log Transformed", out = latex_filename)

################################################################################
# Full dataset for v2x_clpriv including democracies
# Private civil liberties index
# log transformed plot per_mil by v2x_clpriv

plot_clpriv <- ggplot(df, aes(x = v2x_clpriv, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Private Civil Liberties Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Private Civil Liberties Score \nAll Countries Including Democracies") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_clpriv

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Private Civil Liberties Score All Countries Including Democracies.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_clpriv, width = 10, height = 6, dpi = 300)

# multiply by 100
df$v2x_clpriv_100 <- 100 * df$v2x_clpriv

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_clpriv_100,
  data = df
)

summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_13_CLPriv_All_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 13: Private Civil Liberties Score All Countries No Covariates Log Transformed", out = latex_filename)

# only autocracies
df_aut_filtered_clpriv <- df %>% filter(v2x_clpriv < 0.5)

# log transformed plot per_mil by v2x_clpriv
plot_clpriv_aut <- ggplot(df_aut_filtered_clpriv, aes(x = v2x_clpriv, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Private Civil Liberties Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Private Civil Liberties Score \nAuthoritarian Countries Only") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_clpriv_aut

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Private Civil Liberties Score Authoritarian Countries Only.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_clpriv_aut, width = 10, height = 6, dpi = 300)

# multiply by 100
df_aut_filtered_clpriv$v2x_clpriv_100 <- 100 * df_aut_filtered_clpriv$v2x_clpriv

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_clpriv_100,
  data = df_aut_filtered_clpriv
)

summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_14_CLPriv_Aut_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 14: Private Civil Liberties Score Authoritarian Countries No Covariates Log Transformed", out = latex_filename)

################################################################################
# Full dataset for v2x_freexp_altinf including democracies
# Freedom of Expression and Alternative Sources of Information index
# log transformed plot per_mil by v2x_freexp_altinf

plot_freexp_altinf <- ggplot(df, aes(x = v2x_freexp_altinf, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Freedom of Expression and Alternative Sources of Information Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Freedom of Expression and Alternative Sources of Information Score \nAll Countries Including Democracies") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_freexp_altinf

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Freedom of Expression and Alternative Sources of Information Score All Countries Including Democracies.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_freexp_altinf, width = 10, height = 6, dpi = 300)

# multiply by 100
df$v2x_freexp_altinf_100 <- 100 * df$v2x_freexp_altinf

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_freexp_altinf_100,
  data = df
)

summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_15_Freexp_All_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 15: Freedom of Expression and Alternative Sources of Information Score All Countries No Covariates Log Transformed", out = latex_filename)

# only autocracies
df_aut_filtered_freexp <- df %>% filter(v2x_freexp_altinf < 0.5)

# log transformed plot per_mil by v2x_freexp_altinf
plot_freexp_aut <- ggplot(df_aut_filtered_freexp, aes(x = v2x_freexp_altinf, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Freedom of Expression and Alternative Sources of Information Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Freedom of Expression and Alternative Sources of Information Score \nAuthoritarian Countries Only") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_freexp_aut

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Freedom of Expression and Alternative Sources of Information Score Authoritarian Countries Only.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_freexp_aut, width = 10, height = 6, dpi = 300)

# multiply by 100
df_aut_filtered_freexp$v2x_freexp_altinf_100 <- 100 * df_aut_filtered_freexp

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_freexp_altinf_100,
  data = df_aut_filtered_freexp
)

summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_16_Freexp_Aut_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 16: Freedom of Expression and Alternative Sources of Information Score Authoritarian Countries No Covariates Log Transformed", out = latex_filename)

################################################################################
# Full dataset for v2x_frassoc_thick including democracies
# Freedom of association thick index
# log transformed plot per_mil by v2x_frassoc_thick

plot_frassoc_thick <- ggplot(df, aes(x = v2x_frassoc_thick, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Freedom of Association Thick Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Freedom of Association Thick Score \nAll Countries Including Democracies") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_frassoc_thick

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Freedom of Association Thick Score All Countries Including Democracies.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_frassoc_thick, width = 10, height = 6, dpi = 300)

# multiply by 100
df$v2x_frassoc_thick_100 <- 100 * df$v2x_frassoc_thick

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_frassoc_thick_100,
  data = df
)

summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_17_Frassoc_All_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 15: Freedom of Association Thick Score All Countries No Covariates Log Transformed", out = latex_filename)

# only autocracies
df_aut_filtered_frassoc <- df %>% filter(v2x_frassoc_thick < 0.5)

# log transformed plot per_mil by v2x_frassoc_thick
plot_frassoc_thick_aut <- ggplot(df_aut_filtered_frassoc, aes(x = v2x_frassoc_thick, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Freedom of Association Thick Score", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Freedom of Association Thick Score \nAuthoritarian Countries Only") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_frassoc_thick_aut

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Freedom of Association Thick Score Authoritarian Countries Only.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_frassoc_thick_aut, width = 10, height = 6, dpi = 300)

# multiply by 100
df_aut_filtered_frassoc$v2x_frassoc_thick_100 <- 100 * df_aut_filtered_frassoc$v2x_frassoc_thick

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2x_frassoc_thick_100,
  data = df_aut_filtered_frassoc
)

summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_18_Frassoc_Aut_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 18: Freedom of Association Thick Score Authoritarian Countries No Covariates Log Transformed", out = latex_filename)

################################################################################
# Full dataset for v2xcs_ccsi including democracies
# Core civil society index
# log transformed plot per_mil by v2xcs_ccsi

plot_ccsi <- ggplot(df, aes(x = v2xcs_ccsi, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Core Civil Society Index", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Core Civil Society Index \nAll Countries Including Democracies") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_ccsi

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Core Civil Society Index All Countries Including Democracies.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_ccsi, width = 10, height = 6, dpi = 300)

# multiply by 100
df$v2xcs_ccsi_100 <- 100 * df$v2xcs_ccsi

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2xcs_ccsi_100,
  data = df
)

summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_19_CCSI_All_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 19: Core Civil Society Index All Countries No Covariates Log Transformed", out = latex_filename)

# only autocracies
df_aut_filtered_ccsi <- df %>% filter(v2xcs_ccsi < 0.5)

# log transformed plot per_mil by v2xcs_ccsi
plot_ccsi_aut <- ggplot(df_aut_filtered_ccsi, aes(x = v2xcs_ccsi, y = log_per_mil)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Core Civil Society Index", y = "Log of Protests per Million People") +
  ggtitle("Log of Protests per Million People by V-Dem Core Civil Society Index \nAuthoritarian Countries Only") +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  ) +
  theme_bw()

plot_ccsi_aut

# Save the plot as an image
image_name <- "Log of Protests per Million People by V-Dem Core Civil Society Index Authoritarian Countries Only.png"
ggsave(filename = paste0(table_path, image_name), plot = plot_ccsi_aut, width = 10, height = 6, dpi = 300)

# multiply by 100
df_aut_filtered_ccsi$v2xcs_ccsi_100 <- 100 * df_aut_filtered_ccsi$v2xcs_ccsi

# log transformed lin reg
model_poly_log <- lm(
  log_per_mil ~ v2xcs_ccsi_100,
  data = df_aut_filtered_ccsi
)

summary(model_poly_log, digits = 3)
latex_filename <- paste0(table_path, "Model_20_CCSI_Aut_Countries_no_Covariates_Log_Transformed.tex")
stargazer(model_poly_log, type = "latex", title = "Model 20: Core Civil Society Index Authoritarian Countries No Covariates Log Transformed", out = latex_filename)


################################################################################
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
  ) +
  theme_bw()

# Display the plot
lin_plot_poly_aut

# Save the plot as an image
image_name <- "Protests per Million People by V-Dem Polyarchy Score of Authoritarian Countries.png"
ggsave(filename = paste0(table_path, image_name), plot = lin_plot_poly_aut, width = 10, height = 6, dpi = 300)

################################################################################
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
  ) +
  theme_bw()

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
  ) +
  theme_bw()

# Display the plot
lin_plot_poly_aut_sub


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
