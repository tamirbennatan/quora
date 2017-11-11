# Create samples to be answered by humans, for gold standard baseline. 
# Intention is to get 5 subjects to disambiguate 20-40 pairs, for a total of 100-200 human responses. 
# files will be written to seperate CSV files, to be sent to each subject, along with experiment instructions. 

library(dplyr)
library(openxlsx)

# set working directory
setwd("~/Desktop/quora/")

# Load processed data
all.data = read.csv("data/processed/train.csv")

# get 5 different random samples of 40 questions (the samples do not overlap). 
# Only keep the question pairs, the Row ID, and an empty column for the users response
set.seed(550)
sample.data = sample_n(all.data, 1000) %>% 
      select(id, question1, question2) %>%
      mutate(human.response = "")

for (i in 1:25){
      file.name = paste("human_responses/sample", i, ".xlsx", sep = "")
      sheet.name = paste("sample",i)
      write.xlsx(file = file.name, sheetName = sheet.name, x = sample.data[((i-1)*40+1):(i*40),])
}

# Remove objects from memory
rm(sample.data)

