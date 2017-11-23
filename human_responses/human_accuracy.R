library(dplyr)
library(openxlsx)

setwd("~/Desktop/quora/human_responses/responses/")

# get all the responses in a single dataframe
files <- dir()
responses <- do.call(rbind,lapply(files,read.xlsx))

# Get the true values
responses = responses %>% inner_join(
      read.csv("../../data/train.csv") %>% 
            select(id, is_duplicate)
      , by = c("id" = "id"))

# Get the human response accuracy
responses = responses %>% 
      mutate(human.correct = human.response == is_duplicate, 
	group = floor(row_number()/40))
      
human.accuracy = mean(responses$human.correct)
print(paste("Human Accuracy on ", nrow(responses), " responses: ", human.accuracy, sep = ""))

responses %>% group_by(group) %>%
	summarize(acc = mean(human.correct))