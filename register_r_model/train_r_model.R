install.packages("e1071")

library(e1071)
attach(iris)


setwd("C:/Users/antonslutsky/Dev/azureml-quickstart/register_r_model")

MODEL_SAVE_PATH = "./svm_model/artifact"
DEP_LIBS = c("e1071")

x <- subset(iris, select=-Species)
y <- Species

# save model
model <- svm(x,y)

# save
model_rds_path = paste(MODEL_SAVE_PATH, ".rds",sep='')
model_dep_path = paste(MODEL_SAVE_PATH, ".dep",sep='')


# save model
dir.create(dirname(model_rds_path), showWarnings=FALSE, recursive=TRUE)
saveRDS(model, model_rds_path)

# save dependency list
file_conn <- file(model_dep_path)
writeLines(DEP_LIBS, file_conn)
close(file_conn)


model2 = readRDS(model_rds_path)

predict(model2, x)

x[117,]        

v1 = c(5.1, 3.5, 1.4, 0.2)
v2 = c(5.9, 3.0, 5.1, 1.8)
result = array(c(v1, v2), dim = c(4,2))
print(result)

df = t(data.frame(result))

df

predict(model2, df)

head(x)
