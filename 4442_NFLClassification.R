# Data Wrangling
library(data.table)
library(dplyr)
library(fastDummies)
library(magrittr)
library(Matrix)
library(tidyverse)

# Data Visualization
library(DiagrammeR)
library(ggplot2)
library(rpart.plot)
library(rsvg)

# Model-building
library(caret)
library(gbm)
library(parsnip)
library(party)
library(rpart)
library(tidymodels)
library(tree)
library(xgboost)


# Read abbreviated CSV file into local dataframe
nfl <- read.csv("nfl_truncated.csv", header=TRUE, sep=",", fileEncoding="UTF-8-BOM") 

# Filter seasons of interest (based on game_id)
nfl <- nfl %>% filter(game_id > 2015000000)

# Filter out edge-cases (kickoff plays, penalties, timeouts)
nfl <- nfl[nfl$play_type %in% c("run", "pass", "field_goal", "punt"),]

# Convert categorical variables into factors
nfl$play_type <- as.factor(nfl$play_type)
nfl$down <- as.factor(nfl$down)

# Keep only columns of interest 
nfl <- subset(nfl, select=c(
  game_id,  # to be removed later (need it for splitting)
  play_type, 
  half_seconds_remaining, 
  down, 
  ydstogo, 
  yardline_100, 
  score_differential, 
  posteam_timeouts_remaining
))

# Remove null rows
nfl <- na.omit(nfl)

head(nfl)

# Examine the proportion of each play type in the dataset
prop.table(table(nfl$play_type))  

# Countplot of outcome variable (play_type) grouped by down
ggplot(nfl, aes(x=play_type)) + 
  geom_bar(aes(color=down, fill=down)) + 
  ggtitle("Frequency of Play Types based on Down")

# Plot distribution of yards till first down for all the plays
ggplot(nfl, aes(ydstogo)) + geom_bar() + ggtitle("Distribution of Yards-Till-First-Down")

# Plot probability densities of groups of yards till first down with customized breaks to determine best groupings
hist(nfl$ydstogo, col='blue', breaks = c(0, 3, 8, 50))

# Plot distribution broken down by play_type
ggplot(nfl, aes(ydstogo)) + geom_bar(aes(color=play_type, fill=play_type)) + ggtitle("Distribution of Yards-Till-First-Down")

# Plot distribution of yards till first down for all the plays
ggplot(nfl, aes(yardline_100)) + geom_bar() + ggtitle("Distribution of Yards-Till-Endzone")

ggplot(nfl, aes(yardline_100)) + geom_bar(aes(color=play_type, fill=play_type)) + ggtitle("Distribution of Yards-Till-Endzone")

ggplot(nfl, aes(half_seconds_remaining)) + geom_bar() + ggtitle("Distribution of Time Remaining in Half")


# Plot distribution of yards till first down for all the plays
ggplot(nfl, aes(score_differential)) + geom_bar() + ggtitle("Distribution of Score-Differential for all Plays")

# Plot probability densities of groups of yards till first down with customized breaks to determine best groupings
hist(nfl$score_differential, col='red', breaks = c(-50, -14, -7, 0, 7, 14, 50))


# Create box-plots to see range of predictor variables
temp_df <- subset(nfl, select = c(game_id, play_type, down, half_seconds_remaining, ydstogo, yardline_100, score_differential, posteam_timeouts_remaining))

temp_df %>% pivot_longer(half_seconds_remaining:score_differential, names_to="stat", values_to = 'value') %>%
  ggplot(aes(play_type, value, fill=play_type, color=play_type)) +
  geom_boxplot(alpha = 0.25) +
  facet_wrap(~stat, scales = "free_y", nrow = 2) +
  labs(y = NULL, color = NULL, fill = NULL)

temp_df %>% pivot_longer(half_seconds_remaining:score_differential, names_to="stat", values_to = 'value') %>%
  ggplot(aes(play_type, value, fill=down, color=down)) +
  geom_boxplot(alpha = 0.25) +
  facet_wrap(~stat, scales = "free_y", nrow = 2) +
  labs(y = NULL, color = NULL, fill = NULL)

# Create columns containing stratified, factor-versions of numeric variables (to do: determine which col version to use for models)
nfl$yds_factor[nfl$ydstogo <= 4] <- "short"
nfl$yds_factor[nfl$ydstogo > 4 & nfl$ydstogo < 8] <- "med"
nfl$yds_factor[nfl$ydstogo >= 8] <- "long"
nfl$yds_factor <- as.factor(nfl$yds_factor)

nfl$score_factor[nfl$score_differential < -7] <- "down_big"
nfl$score_factor[nfl$score_differential >= -7 & nfl$score_differential < 0] <- "down_score"
nfl$score_factor[nfl$score_differential == 0] <- "tied"
nfl$score_factor[nfl$score_differential > 0 & nfl$score_differential <= 7] <- "up_score"
nfl$score_factor[nfl$score_differential > 7] <- "up_big"
nfl$score_factor <- as.factor(nfl$score_factor)

str(nfl)


# Transform factor-type predictor variables into dummy-vectors (for model-fitting)
nfl <- dummy_cols(
  nfl,
  select_columns = c("down", "yds_factor", "score_factor"),
  remove_selected_columns = TRUE,
  remove_first_dummy = TRUE  # to avoid multi-collinearity issues in model fit
) 

str(nfl)


# Split columns of interest into training & testing sets with desired specs
full_data <- subset(nfl, select=c(
  game_id,
  play_type, 
  half_seconds_remaining, 
  down_2,
  down_3,
  down_4,
  ydstogo, 
  yardline_100, 
  score_differential, 
  posteam_timeouts_remaining
))

train <- full_data[full_data$game_id < 2018000000,]  # train using 2015-2017 data
test <- full_data[full_data$game_id > 2018000000,]  # test using 2018 data
train = subset(train, select = -c(game_id))  # remove game_id 
test = subset(test, select = -c(game_id))  # remove game_id


## DECISION-TREE WITH TREE PKG (BASELINE)

# Copy split sets to avoid cross-contamination
train_tree <- train
test_tree <- test

dtime <- Sys.time()  # time this model
tree_model = tree(play_type ~ ., data=train_tree)
dtime <- Sys.time() - dtime

tree_pred = predict(tree_model, test_tree[,-1], type="class")
tree_confmat <- table(PredictedPlays=tree_pred, ActualPlays=test_tree$play_type)
tree_confmat

# Compute accuracy
tree_acc <- sum(diag(tree_confmat))/sum(tree_confmat)
tree_acc

# Compute accuracy broken down by play types
df <- data.frame(pred=tree_pred, actual=test_tree$play_type)
fg <- df[df$actual %in% c("field_goal"),]
confusionMatrix(as.factor(fg$pred), fg$actual)
punts <- df[df$actual %in% c("punt"),]
confusionMatrix(as.factor(punts$pred), punts$actual)
pass <- df[df$actual %in% c("pass"),]
confusionMatrix(as.factor(pass$pred), pass$actual)
runs <- df[df$actual %in% c("run"),]
confusionMatrix(as.factor(runs$pred), runs$actual)


# DECISION-TREE WITH RPART PKG (More substantial version, to generate a cleaner example visual of decision-tree of NFL data)

# Copy split sets to avoid cross-contamination
train_rpart <- train
test_rpart <- test

# Execute decision-tree algorithm with custom specifications
tree_rpart <- rpart(
  play_type ~ ., 
  cp=0.001,  # complexity parameter (lower means more complex tree)
  maxdepth=5,  # maximum tree depth
  minbucket=2000,  # min num of observations in lead nodes
  method="class",  # classfication instead of regression probabilities
  data=train_rpart 
)

options(repr.plot.width=14, repr.plot.height=8)
rpart.plot(tree_rpart, extra=4)  # extra gives us probs for each outcome 
prp(tree_rpart)


# Copy split sets to avoid cross-contamination
train_gbm <- train
test_gbm <- test

btime <- Sys.time() # time this model
# Fit gradient-boosted model for training dataset
boost <- gbm(
  play_type ~ ., 
  data=train_gbm, 
  distribution="multinomial", 
  n.trees=50,  # higher => slower; make sure to match with num trees in xgboost for proper comparison
  shrinkage=0.05,
  bag.fraction=0.5,
  cv.folds=5,
  keep.data=FALSE,
  interaction.depth=4
)
btime <- Sys.time() - btime

# Ideal amount of trees
ntrees <- gbm.perf(boost)
ntrees

boost
summary(boost)



# Evaluate the gradient-boosting model
boost_probs <- predict(boost, newdata=test_gbm, type="response", n.trees=ntrees)  # gives us probabilities, not predicted classification
boost_pred <- matrix(boost_probs, nrow=nrow(boost_probs), ncol=4) %>% 
  data.frame() %>%
  mutate(max_prob=max.col(., "last")-1)  # grabs the column with highest probability (predicted classification)

# Manually recode values for highest probabilities to be able to compare in confusion matrix
boost_pred$max_prob[boost_pred$max_prob == 0] <- "field_goal"
boost_pred$max_prob[boost_pred$max_prob == 1] <- "pass"
boost_pred$max_prob[boost_pred$max_prob == 2] <- "punt"
boost_pred$max_prob[boost_pred$max_prob == 3] <- "run"

# Evaluate performance of model
confusionMatrix(as.factor(boost_pred$max_prob), test_gbm$play_type)

# Compute accuracies broken down by play type
df <- data.frame(pred=boost_pred$max_prob, actual=test_gbm$play_type)
fg <- df[df$actual %in% c("field_goal"),]
confusionMatrix(as.factor(fg$pred), fg$actual)
punts <- df[df$actual %in% c("punt"),]
confusionMatrix(as.factor(punts$pred), punts$actual)
pass <- df[df$actual %in% c("pass"),]
confusionMatrix(as.factor(pass$pred), pass$actual)
runs <- df[df$actual %in% c("run"),]
confusionMatrix(as.factor(runs$pred), runs$actual)


# Copy split sets to avoid cross-contamination
train_xg <- train
test_xg <- test

# Prepare matrices in appropriate format for to feed into xgboost algorithm
train_xg_mat <- xgb.DMatrix(
  data = model.matrix(play_type ~ .-1, data = train_xg), 
  label = as.numeric(train_xg[,"play_type"])-1
) 
test_xg_mat <- xgb.DMatrix(
  data = model.matrix(play_type ~ .-1, data = test_xg), 
  label = as.numeric(test_xg[,"play_type"])-1
)  

# Setup parameters for xg model
xgb_params <- list(
  objective = "multi:softprob",  # since we have multiple categories
  eval_metric = "mlogloss",  # 
  num_class= 4,  # number of outcome categories
  eta = .1,  # learning rate (lower eta prevents overfitting)
  max.depth = 5,  # maximum tree depth (complexity of tree)
  gamma = 0,  # larger gamma means more conservative algorithm (to avoid overfitting)
  subsample = 1,  # proportion of data to be used for growing trees (lower minimizes overfitting risk)
  min_child_weight = 0.8,  # proportion of weight to put on the important predictor variables
  colsample_bytree = 1,  # proportion of predictor variables to look at (?) to prevent collinearity
  booster = "gbtree"  # tree-based algorithm
)
watchlist <- list(train=train_xg_mat, test=test_xg_mat)  # for observing step by step iterations

xtime <- Sys.time() # track time for model training
xg <- xgb.train(params = xgb_params,
                data = train_xg_mat,
                nrounds = 50,  # num of iterations  (can change based on overfitting and what iteration has min based on code below)
                watchlist = watchlist  # to output steps 
)
xtime <- Sys.time() - xtime

xg

# Training & test error plot
e <- data.frame(xg$evaluation_log)  # storing errors
plot(e$iter, e$train_mlogloss, col = 'blue', main="Train vs Test LogLoss Over Each Iteration")
lines(e$iter, e$test_mlogloss, col = 'red') 


# Determine the most important predictor variables for our model
imp <- xgb.importance(colnames(train_xg_mat), model = xg)  # independent variables with most gain
imp

imp_df <- imp[imp$Feature %in% c("down_4", "yardline_100", "ydstogo", "half_seconds_remaining", "down_3", "score_differential"),]
importance_plot <- ggplot(imp, aes(x=reorder(Feature, Gain), y=Gain, fill=Gain)) +
  scale_fill_gradient2(low="yellow", mid="orange", high="red", space="Lab", guide = "colourbar") +
  geom_bar(stat='identity') +
  coord_flip() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank()) +
  
  labs(title = "Feature Importance from XG-Boost", x = "Feature", y = "Gain") + theme(legend.position="none")
importance_plot



# Restructure computed predictions & evaluate xgboost model
xg_probs <- predict(xg, newdata=test_xg_mat, type="class")  # probs 
xg_pred <- matrix(xg_probs, ncol=length(xg_probs)/4, nrow=4) %>%
  t() %>%  # transpose
  data.frame() %>%
  mutate(max_prob=max.col(., "last")-1)  # grabs the column with highest probability (predicted classification)

# Manually recode to original values for highest probabilities to be able to compare in confusion matrix
xg_pred$max_prob[xg_pred$max_prob == 0] <- "field_goal"
xg_pred$max_prob[xg_pred$max_prob == 1] <- "pass"
xg_pred$max_prob[xg_pred$max_prob == 2] <- "punt"
xg_pred$max_prob[xg_pred$max_prob == 3] <- "run"

confusionMatrix(as.factor(xg_pred$max_prob), test_xg$play_type)


# Evaluate model broken down by play
df <- data.frame(pred=xg_pred$max_prob, actual=test_xg$play_type)
fg <- df[df$actual %in% c("field_goal"),]
confusionMatrix(as.factor(fg$pred), fg$actual)
punts <- df[df$actual %in% c("punt"),]
confusionMatrix(as.factor(punts$pred), punts$actual)
pass <- df[df$actual %in% c("pass"),]
confusionMatrix(as.factor(pass$pred), pass$actual)
runs <- df[df$actual %in% c("run"),]
confusionMatrix(as.factor(runs$pred), runs$actual)


# Plot the accuracies for each play
df2 <- data.frame(
  play_type=c("field-goal", "punt", "pass", "run"),
  accuracy=c(91.79, 98.27, 72.18, 61.1),
  ymin=c(89.79, 97.6, 71.51, 60.22),
  ymax=c(93.51, 98.79, 72.84, 61.98))

g <- ggplot(df2, aes(x=play_type, y=accuracy, color=play_type, label=accuracy)) +
  geom_pointrange(aes(ymin=ymin, ymax=ymax)) + geom_point(size = 2) +
  geom_text(aes(label=accuracy), vjust=-0.25, color = "black", size=3.5, nudge_x=0, nudge_y=1.25, show.legend=NA)
g


# Plot projected tree accumulating results from all iterative models
xgb.plot.multi.trees(
  model=xg, 
  feature_names=c("down_4", "down_3", "down_2", "ydstogo", "yardline_100", "half_seconds_remaining", "score_differential", "pos_team_timeouts_remaining"),
  # features.keep=2,
  plot_width=500, 
  render=TRUE,
  plot_height=800
)


# Plot model complexity
xgb.plot.deepness(model = xg)


# # set up the cross-validated hyper-parameter search
# xgb_grid_1 = expand.grid(
# nrounds = 50,
# eta = c(0.1, 0.5, 0.05),
# max_depth = c(10, 15),
# gamma = c(0.5, 0.75),
# subsample = 1,  
# min_child_weight = 1,  
# colsample_bytree = 1
# )
# 
# # pack the training control parameters
# xgb_trcontrol_1 = trainControl(
# method = "cv",
# number = 5,
# verboseIter = TRUE,
# returnData = FALSE,
# returnResamp = "all",                                                     
# classProbs = TRUE,                                                           
# # summaryFunction = twoClassSummary,
# allowParallel = TRUE
# )
# 
# # train the model for each parameter combination in the grid,
# #   using CV to evaluate
# xgb_train_1 = train(
# x = as.matrix(train_xg %>%
# select(-play_type)),
# y = as.factor(train_xg$play_type),
# trControl = xgb_trcontrol_1,
# tuneGrid = xgb_grid_1,
# method = "xgbTree"
# )


# Plot grid containing accuracies and times of explored models
methods <- c("Decision-Tree", "Gradient-Boosting", "XG-Boost")
runtime_seconds <- c(as.numeric(dtime), as.numeric(btime), as.numeric(xtime))
accuracy_rate <- c(62.64, 69.01, 70.25)
compare_df <- data.frame(methods, runtime_seconds, accuracy_rate)

compare_plot <- ggplot(compare_df, aes(x=runtime_seconds, y=accuracy_rate, color=methods)) +
  geom_point(size=5) +
  lims(x=c(0,40),y=c(55, 75)) +
  theme_minimal() +
  coord_fixed() +  
  geom_vline(xintercept = 20) + geom_hline(yintercept = 65) +
  # geom_label(aes(label = methods), color = "black", size=3, nudge_x=0, nudge_y=2, show.legend=NA) +
  geom_text(aes(label = methods), color = "black", size=3.5, nudge_x=0, nudge_y=1.25, show.legend=NA) +
  labs(title = "Comparing Methods", x = "Runtime (in seconds)", y = "Accuracy (%)") + theme(legend.position="none")
compare_plot
