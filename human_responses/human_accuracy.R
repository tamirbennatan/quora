library(dplyr)
library(openxlsx)
library(caret)

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
# accuracy      
human.accuracy = mean(responses$human.correct)

# get a confusion matrix
cm = confusionMatrix(responses$human.response, responses$is_duplicate)

# precision, recall and f1 
precision = cm$byClass['Precision']
recall = cm$byClass['Recall']
f1.score = cm$byClass['F1']

# print output
print(paste("Human Accuracy on ", nrow(responses), " responses: ", human.accuracy, sep = ""))
print(paste("Human Precision on  ", nrow(responses), " responses: ", precision, sep = ""))
print(paste("Human Recall on ", nrow(responses), " responses: ", recall, sep = ""))
print(paste("Human F1 Score on ", nrow(responses), " responses: ", f1.score, sep = ""))


# write to a text file
write.csv(c("Accuracy" = human.accuracy, "Precision" = precision, "Recall" = recall), 
          "~/Desktop/quora/human_responses/results.txt")

