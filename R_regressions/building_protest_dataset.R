library(ggplot2)
library(dplyr)

# get filenames of all .tab files in "./exp data" directory
files <- list.files(path = "./exp data", pattern = "*", full.names = TRUE)

# initialize an empty list to store the data frames
df_list <- list()

# loop through each file and read in the data, and print the filename
for (file in files) {
  print(paste("Loading file", file))
  df_list[[file]] <- read.delim(file, sep = "\t", header = TRUE)
}
# bind all the data frames together
data <- do.call(rbind, df_list)

dim(data)
head(data)
names(data)

data_14 <- data[grepl("^14", data$CAMEO.Code), ]

head(data_14)
dim(data_14)
unique(data_14$CAMEO.Code)

write.csv(data_14, "protests.csv", row.names = FALSE)
