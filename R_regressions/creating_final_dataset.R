library(ggplot2)
library(dplyr)
library(tidyverse)
library(lubridate)
library(knitr)
library(naniar)
library(stargazer)


# load in protests.csv
protests <- read.csv("protests.csv")
full_data_dim <- dim(protests)
names(protests)
unique(protests$Country)
full_data_dim

# inspect the data
head(protests)

# examining empty and null values in the Country column
protests_empty_country <- protests %>%
  filter(Country == "")

head(protests_empty_country)
dim(protests_empty_country)

protest_filtered <- protests[protests$Country != "", ]

protests_null_country <- protests %>%
  filter(Country == "NULL")

head(protests_null_country)
dim(protests_null_country)

protest_filtered <- protest_filtered[protest_filtered$Country != "NULL", ]

unique(protest_filtered$Country)

protests_na_country <- protests %>%
  filter(is.na(Country))

head(protests_na_country)
dim(protests_na_country)

# remove rows with no country data
dim(protests)
protest_filtered <- protest_filtered[!is.na(protest_filtered$Country), ]
dim(protest_filtered) - full_data_dim

# create a table of the number of protests per country sorted by number of protests
protest_count <- protest_filtered %>%
  group_by(Country) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count))

protest_count

# count NAs in Event.Date
sum(is.na(protest_filtered$Event.Date))

# create a new column for year of protest from the date column
na_year_dates <- protest_filtered
na_year_dates$temp <- ymd(protest_filtered$Event.Date)
na_year_dates$year <- year(na_year_dates$temp)

# count NAs in the new Year column
sum(is.na(na_year_dates$year))

# use MDY to parse dates with no year
na_year_dates$temp1 <- mdy(na_year_dates$Event.Date)
# switch date scheme from MDY to YMD
na_year_dates$temp2 <- ymd(na_year_dates$temp1)
na_year_dates$Year[is.na(na_year_dates$Year)] <- year(na_year_dates$temp1[is.na(na_year_dates$Year)])
sum(is.na(na_year_dates$Year))

# assign Year column to protest_filtered
protest_filtered$year <- na_year_dates$year
head(protest_filtered)

# drop 2022 because of population issues
protest_filtered <- protest_filtered[protest_filtered$year != 2022, ]

# change name of protest_filtered Year column to year
colnames(protest_filtered)[18] <- "country"

# drop unneeded columns in protest_filtered
protest_filtered <- protest_filtered[, c(18, 21)]

# load V-Dem data
vdem <- read.csv("Vdem/Country_Year_V-Dem_Core_CSV_v12/V-Dem-CY-Core-v12.csv")

# remove all row data with year before 1995
vdem <- vdem[vdem$year >= 1995, ]

# change vdem country_name to Country and year to Year
colnames(vdem)[1] <- "country"

# Get unique countries in pop
vdem_countries <- unique(vdem$country)

# Get unique countries in protest_filtered
protest_countries <- unique(protest_filtered$country)

# Find countries in merged_summ that are not in vdem
protest_only <- setdiff(protest_countries, vdem_countries)

# Find countries in vdem that are not in merged_summ
vdem_only <- setdiff(vdem_countries, protest_countries)

print(sort(vdem_only))
print(sort(protest_only))

# change country names in protest filtered to match vdem where possible
protest_filtered$country[protest_filtered$country == "Myanmar"] <- "Burma/Myanmar"
protest_filtered$country[protest_filtered$country == "Democratic Republic of Congo"] <- "Democratic Republic of the Congo"
protest_filtered$country[protest_filtered$country == "Moldova, Republic of"] <- "Moldova"
protest_filtered$country[protest_filtered$country == "Russia"] <- "North Macedonia"
protest_filtered$country[protest_filtered$country == "Occupied Palestinian Territory"] <- "Palestine/West Bank"
protest_filtered$country[protest_filtered$country == "Congo"] <- "Republic of the Congo"
protest_filtered$country[protest_filtered$country == "Russian Federation"] <- "Russia"
protest_filtered$country[protest_filtered$country == "the former Yugoslav Republic of Macedonia"] <- "North Macedonia"
protest_filtered$country[protest_filtered$country == "United States"] <- "United States of America"
protest_filtered$country[protest_filtered$country == "Gambia"] <- "The Gambia"

# check if successful
# Get unique countries in pop
vdem_countries <- unique(vdem$country)

# Get unique countries in protest_filtered
protest_countries <- unique(protest_filtered$country)

# Find countries in merged_summ that are not in vdem
protest_only <- setdiff(protest_countries, vdem_countries)

# Find countries in vdem that are not in merged_summ
vdem_only <- setdiff(vdem_countries, protest_countries)

print(sort(vdem_only))
print(sort(protest_only))

# list of countries not in vdem and lost rows in removing those
protest_only <- setdiff(protest_countries, vdem_countries)
ori_rows <- nrow(protest_filtered)

# drop countries in protest filtered in list of protest_only
protest_filtered <- protest_filtered[!protest_filtered$country %in% protest_only, ]

# remove all columns except the wanted columns
vdem <- vdem[, c("country", "country_text_id", "year", "v2x_polyarchy", "v2x_civlib", "v2x_clpol", "v2x_clpriv", "v2x_freexp_altinf", "v2x_frassoc_thick", "v2xcl_disc", "v2x_freexp", "v2xcs_ccsi")]

# Create a data frame of all possible country-year combinations
all_country_years <- expand.grid(
  country = unique(protest_filtered$country),
  year = unique(protest_filtered$year)
)

counted_protest <- protest_filtered %>%
  group_by(country, year) %>%
  summarize(count = n()) %>%
  ungroup()

# Left join the summarized data frame with the full set of country-year combinations
complete_counted_df <- all_country_years %>%
  left_join(counted_protest, by = c("country", "year")) %>%
  replace_na(list(count = 0)) # Replace NAs in the count column with 0

head(complete_counted_df)
dim(complete_counted_df)

# add vdem data to protest_filtered by country and year
merged_df <- complete_counted_df %>%
  left_join(vdem, by = c("country", "year"))

# drop Taiwan from merged_df
merged_df <- merged_df[merged_df$country != "Taiwan", ]

# drop all rows with NA values in country_text_id
merged_df <- merged_df[!is.na(merged_df$country_text_id), ]

# load population data
# https://data.worldbank.org/indicator/SP.POP.TOTL?locations=VC
pop <- read.csv("world bank data/API_SP.POP.TOTL_DS2_en_csv_v2_4770387.csv")

head(pop)

# remove the X before the year for each column
colnames(pop)[grepl("^X", colnames(pop))] <- sub("^X", "", colnames(pop)[grepl("^X", colnames(pop))])
head(pop)

# change the column name of Country.Name to Country
colnames(pop)[1] <- "country"
colnames(pop)[2] <- "country_text_id"

# check which countries are in the protest data but not in the population data
# Get unique countries in pop
pop_countries <- unique(pop$country_text_id)

# Get unique countries in protest_filtered
protest_countries <- unique(complete_counted_df$country_text_id)

# Find countries in protest_filtered that are not in pop
protest_only <- setdiff(protest_countries, pop_countries)

# Find countries in pop that are not in protest_filtered
pop_only <- setdiff(pop_countries, protest_countries)

print(sort(pop_only))
print(sort(protest_only))

# merge population data with protest data
head(pop)

pop_tidy <- pop %>%
  pivot_longer(
    cols = -c(country, country_text_id, Indicator.Name, Indicator.Code),
    names_to = "year",
    values_to = "population"
  ) %>%
  mutate(year = as.integer(year))

colnames(pop_tidy)[5] <- "year"
pop_tidy <- pop_tidy[, -c(3, 4)]

# drop pop_tidy country column
pop_tidy <- pop_tidy[, -1]

# pop_tidy stats
dim(pop_tidy)

# remove all years before 1995 and after 2021
pop_tidy <- pop_tidy[pop_tidy$year >= 1995 & pop_tidy$year <= 2021, ]
dim(pop_tidy)
head(pop_tidy)
summary(pop_tidy)

# merge pop_tidy with counted_df by country_text_id and year
merged_data <- merged_df %>%
  left_join(pop_tidy, by = c("country_text_id", "year"))

head(merged_data)
dim(merged_data)

summary(merged_data)

# create column for number of protests per million people
merged_data$per_mil <- merged_data$count / (merged_data$population / 1000000)

head(merged_data, 20)

# create column log_per_mil
merged_data$log_per_mil <- log(merged_data$per_mil + 1)

# import world bank data for inflation
inflation <- read.csv("world bank data/API_NY.GDP.DEFL.KD.ZG_DS2_en_csv_v2_4901885.csv")

names(inflation)
colnames(inflation)[grepl("^X", colnames(inflation))] <- sub("^X", "", colnames(inflation)[grepl("^X", colnames(inflation))])
names(inflation)

# change the column name of Country.Name to Country
colnames(inflation)[1] <- "country"
colnames(inflation)[2] <- "country_text_id"

inflation_tidy <- inflation %>%
  pivot_longer(
    cols = -c(country, country_text_id, Indicator.Name, Indicator.Code),
    names_to = "year",
    values_to = "inflation"
  ) %>%
  mutate(year = as.integer(year))

# drop inflation_tidy year < 1995
inflation_tidy <- inflation_tidy[inflation_tidy$year >= 1995, ]

# drop inflation_tidy columns country, Indicator.Name, Indicator.Code
inflation_tidy <- inflation_tidy[, -c(1, 3, 4)]

head(inflation_tidy)

# merge inflation_tidy with merged_data by country_text_id and year
merged_data <- merged_data %>%
  left_join(inflation_tidy, by = c("country_text_id", "year"))

# check merged_data
head(merged_data)

# import world bank data for gdp
gdp <- read.csv("world bank data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4901850.csv")

names(gdp)
colnames(gdp)[grepl("^X", colnames(gdp))] <- sub("^X", "", colnames(gdp)[grepl("^X", colnames(gdp))])
names(gdp)

# change the column name of Country.Name to Country
colnames(gdp)[1] <- "country"
colnames(gdp)[2] <- "country_text_id"
names(gdp)

gdp_tidy <- gdp %>%
  pivot_longer(
    cols = -c(country, country_text_id, Indicator.Name, Indicator.Code),
    names_to = "year",
    values_to = "gdp"
  ) %>%
  mutate(year = as.integer(year))

# drop gdp_tidy year < 1995
gdp_tidy <- gdp_tidy[gdp_tidy$year >= 1995, ]
names(gdp_tidy)
# drop gdp_tidy columns country, Indicator.Name, Indicator.Code
gdp_tidy <- gdp_tidy[, -c(1, 3, 4)]
names(gdp_tidy)

# merge gdp_tidy with merged_data by country_text_id and year
merged_data <- merged_data %>%
  left_join(gdp_tidy, by = c("country_text_id", "year"))

# check merged_data
head(merged_data)

# import world bank data for gdp growth
gdp_growth <- read.csv("world bank data/API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_4901640.csv")

names(gdp_growth)
colnames(gdp_growth)[grepl("^X", colnames(gdp_growth))] <- sub("^X", "", colnames(gdp_growth)[grepl("^X", colnames(gdp_growth))])
names(gdp_growth)

# change the column name of Country.Name to Country
colnames(gdp_growth)[1] <- "country"
colnames(gdp_growth)[2] <- "country_text_id"
names(gdp_growth)

gdp_growth_tidy <- gdp_growth %>%
  pivot_longer(
    cols = -c(country, country_text_id, Indicator.Name, Indicator.Code),
    names_to = "year",
    values_to = "gdp_growth"
  ) %>%
  mutate(year = as.integer(year))

# drop gdp_growth_tidy year < 1995
gdp_growth_tidy <- gdp_growth_tidy[gdp_growth_tidy$year >= 1995, ]
names(gdp_growth_tidy)
# drop gdp_growth_tidy columns country, Indicator.Name, Indicator.Code
gdp_growth_tidy <- gdp_growth_tidy[, -c(1, 3, 4)]
names(gdp_growth_tidy)

# merge gdp_growth_tidy with merged_data by country_text_id and year
merged_data <- merged_data %>%
  left_join(gdp_growth_tidy, by = c("country_text_id", "year"))

# check merged_data
tail(merged_data)

# import world bank data for gdppc
gdppc <- read.csv("world bank data/API_NY.GDP.PCAP.KD_DS2_en_csv_v2_4901640.csv")

names(gdppc)
colnames(gdppc)[grepl("^X", colnames(gdppc))] <- sub("^X", "", colnames(gdppc)[grepl("^X", colnames(gdppc))])
names(gdppc)

# change the column name of Country.Name to Country
colnames(gdppc)[1] <- "country"
colnames(gdppc)[2] <- "country_text_id"
names(gdppc)

gdppc_tidy <- gdppc %>%
  pivot_longer(
    cols = -c(country, country_text_id, Indicator.Name, Indicator.Code),
    names_to = "year",
    values_to = "gdppc"
  ) %>%
  mutate(year = as.integer(year))

# drop gdppc_tidy year < 1995
gdppc_tidy <- gdppc_tidy[gdppc_tidy$year >= 1995, ]
names(gdppc_tidy)
# drop gdppc_tidy columns country, Indicator.Name, Indicator.Code
gdppc_tidy <- gdppc_tidy[, -c(1, 3, 4)]
names(gdppc_tidy)

# merge gdppc_tidy with merged_data by country_text_id and year
merged_data <- merged_data %>%
  left_join(gdppc_tidy, by = c("country_text_id", "year"))

# check merged_data
tail(merged_data)

# import world bank data for gdppc growth
gdppc_growth <- read.csv("world bank data/API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_4901900.csv")

names(gdppc_growth)
colnames(gdppc_growth)[grepl("^X", colnames(gdppc_growth))] <- sub("^X", "", colnames(gdppc_growth)[grepl("^X", colnames(gdppc_growth))])
names(gdppc_growth)

# change the column name of Country.Name to Country
colnames(gdppc_growth)[1] <- "country"
colnames(gdppc_growth)[2] <- "country_text_id"
names(gdppc_growth)

gdppc_growth_tidy <- gdppc_growth %>%
  pivot_longer(
    cols = -c(country, country_text_id, Indicator.Name, Indicator.Code),
    names_to = "year",
    values_to = "gdppc_growth"
  ) %>%
  mutate(year = as.integer(year))

# drop gdppc_growth_tidy year < 1995
gdppc_growth_tidy <- gdppc_growth_tidy[gdppc_growth_tidy$year >= 1995, ]
names(gdppc_growth_tidy)
# drop gdppc_growth_tidy columns country, Indicator.Name, Indicator.Code
gdppc_growth_tidy <- gdppc_growth_tidy[, -c(1, 3, 4)]
names(gdppc_growth_tidy)

# merge gdppc_growth_tidy with merged_data by country_text_id and year
merged_data <- merged_data %>%
  left_join(gdppc_growth_tidy, by = c("country_text_id", "year"))

# check merged_data
tail(merged_data)

# move column 3 to column 1
merged_data <- merged_data[, c(3, 1:2, 4:ncol(merged_data))]
#move column 11 and 12 to 5 and 6
merged_data <- merged_data[, c(1:4, 11:12, 5:10, 13:ncol(merged_data))]

summary(merged_data)

# divide gdppc by 1000
merged_data$gdppc <- merged_data$gdppc / 1000
# Standardize and center the gdp variable
standardized_gdp <- scale(merged_data$gdp)

# Add the standardized variable back to the dataset as a new column
merged_data$standardized_gdp <- standardized_gdp

# lagged variables
# create lagged indicator of gdp_growth, inflation, gdppc_growth
merged_data <- merged_data %>%
    mutate(gdp_growth_lag = lag(gdp_growth, 1)) %>%
    mutate(inflation_lag = lag(inflation, 1)) %>%
    mutate(gdppc_growth_lag = lag(gdppc_growth, 1))

tail(merged_data)


# check merged_data
tail(merged_data)
dim(merged_data)
# # write merged_summ to csv
write.csv(merged_data, "prot_country_year_pop_mil.csv")
