#install.packages("caret")
#install.packages("DALEX")
#install.packages("ggplot2")
#install.packages("randomForest")
#install.packages("kernlab")
#install.packages("pROC")
#install.packages("xgboost")
#install.packages("fs")
dir()          # check file
ls()
rm(list=ls())
#引用包
library(caret)
library(DALEX)
library(ggplot2)
library(randomForest)
library(kernlab)
library(xgboost)
library(pROC)
library(fs)

#创建一个文件夹
dir_create("结果")

#设置种子，重复分析结果
set.seed(123) 

#设置工作目录
setwd("D:/learning/lobectomyablation/radiomic/9.多机器学习算法建模")

# 读取文件
train <- read.csv('train.csv', row.names = 1, check.names = F)
test <- read.csv('test.csv', row.names = 1, check.names = F)
data <- rbind(train,test)

#RF随机森林树模型
control=trainControl(method="repeatedcv", number=5, savePredictions=TRUE)
mod_rf = train(Type ~ ., data = train, method='rf', 
               trControl = control)

#SVM机器学习模型
mod_svm=train(Type ~., data = train, method = "svmRadial",
              prob.model=TRUE, trControl=control)

#GLM模型
mod_glm=train(Type ~., data = train, method = "glm",
              family="binomial", trControl=control)

#GBM模型
mod_gbm=train(Type ~., data = train, method = "gbm", 
              trControl=control)
#KNN模型
mod_knn=train(Type ~., data = train, method = "knn", 
              trControl=control)

#NNET模型
mod_nnet=train(Type ~., data = train, method = "nnet",
               trControl=control)

#Lsso模型
mod_lasso=train(Type ~., data = train, method = "glmnet", 
                trControl=control)

#DT模型
mod_dt=train(Type ~., data = train, method = "rpart",
             trControl=control)

#定义预测函数
p_fun=function(object, newdata){
  predict(object, newdata=newdata, type="prob")[,2]
}
yTrain=ifelse(train$Type=="Control", 0, 1)##注意更改自己的分类类型


###########################test####################
#1 RF随机森林树模型预测结果
explainer_rf=explain(mod_rf, label = "RF",
                     data = train, y = yTrain,
                     predict_function = p_fun,
                     verbose = FALSE)
mp_rf=model_performance(explainer_rf)
#recall     : 1 
#precision  : 1 
#f1         : 1 
#accuracy   : 1 
#auc        : 1

#2 SVM机器学习模型预测结果
explainer_svm=explain(mod_svm, label = "SVM",
                      data = train, y = yTrain,
                      predict_function = p_fun,
                      verbose = FALSE)
mp_svm=model_performance(explainer_svm)
#recall     : 0.7878788 
#precision  : 0.7878788 
#f1         : 0.7878788 
#accuracy   : 0.8738739 
#auc        : 0.9176379

#3 GLM模型预测结果
explainer_glm=explain(mod_glm, label = "GLM",
                      data = train, y = yTrain,
                      predict_function = p_fun,
                      verbose = FALSE)
mp_glm=model_performance(explainer_glm)
#Measures for:  classification
#recall     : 0.7272727 
#precision  : 0.8275862 
#f1         : 0.7741935 
#accuracy   : 0.8738739 
#auc        : 0.9529915

###4 GBM模型预测结果
explainer_gbm=explain(mod_gbm, label = "GBM",
                      data = train, y = yTrain,
                      predict_function = p_fun,
                      verbose = FALSE)
mp_gbm=model_performance(explainer_gbm)
#recall     : 0.7575758 
#precision  : 0.9615385 
#f1         : 0.8474576 
#accuracy   : 0.9189189 
#auc        : 0.9724165

#5 KKNN模型预测结果
explainer_knn=explain(mod_knn, label = "KNN",
                       data = train, y = yTrain,
                       predict_function = p_fun,
                       verbose = FALSE)
mp_knn=model_performance(explainer_knn)
#recall     : 0.6363636 
#precision  : 0.875 
#f1         : 0.7368421 
#accuracy   : 0.8648649 
#auc        : 0.9520202

#6 nnet模型预测结果
explainer_nnet=explain(mod_nnet, label = "NNET",
                       data = train, y = yTrain,
                       predict_function = p_fun,
                       verbose = FALSE)
mp_nnet=model_performance(explainer_nnet)
#recall     : 0.9090909 
#precision  : 0.9677419 
#f1         : 0.9375 
#accuracy   : 0.963964 
#auc        : 0.9965035

#7 lasso模型预测结果
explainer_lasso=explain(mod_lasso, label = "LASSO",
                        data = train, y = yTrain,
                        predict_function = p_fun,
                        verbose = FALSE)
mp_lasso=model_performance(explainer_lasso)
#recall     : 0.7272727 
#precision  : 0.8888889 
#f1         : 0.8 
#accuracy   : 0.8918919 
#auc        : 0.9506605

#8 DT模型预测结果
explainer_dt=explain(mod_dt, label = "DT",
                        data = train, y = yTrain,
                        predict_function = p_fun,
                        verbose = FALSE)
mp_dt=model_performance(explainer_dt)
#recall     : 0.6666667 
#precision  : 0.9166667 
#f1         : 0.7719298 
#accuracy   : 0.8828829 
#auc        : 0.8205128

#绘制四种方法的残差反向累计分布图
#下面的链接解释信息
#https://okan.cloud/posts/2021-03-23-visualizing-machine-learning-models/
pdf(file="结果/绝对残差图.pdf", width=6, height=6)
p1 <- plot(mp_rf, mp_svm, mp_glm, mp_gbm, mp_knn, mp_nnet,mp_lasso,mp_dt)
print(p1)
dev.off()

#绘制四种方法的残差箱线图
pdf(file="结果/箱形图.pdf", width=6, height=6)
p2 <- plot(mp_rf, mp_svm, mp_glm, mp_gbm, mp_knn , mp_nnet, mp_lasso, mp_dt,geom = "boxplot")
print(p2)
dev.off()

p3 <- plot(mp_rf, mp_svm, mp_glm, mp_gbm, mp_knn , mp_nnet, mp_lasso, mp_dt, geom = "histogram") 
print(p3)

p4 <- plot(mp_rf, mp_svm, mp_glm, mp_gbm, mp_knn , mp_nnet, mp_lasso, mp_dt, geom = "prc") 
print(p4)


#绘制ROC曲线
pred1=predict(mod_rf, newdata=train, type="prob")
pred2=predict(mod_svm, newdata=train, type="prob")
pred3=predict(mod_glm, newdata=train, type="prob")
pred4=predict(mod_gbm, newdata=train, type="prob")
pred5=predict(mod_knn, newdata=train, type="prob")
pred6=predict(mod_nnet, newdata=train, type="prob")
pred7=predict(mod_lasso, newdata=train, type="prob")
pred8=predict(mod_dt, newdata=train, type="prob")

roc1=roc(yTrain, as.numeric(pred1[,2]), ci = TRUE)
roc1
#Area under the curve: 1
#95% CI: 1-1 (DeLong)
roc2=roc(yTrain, as.numeric(pred2[,2]), ci = TRUE)
roc2
#Area under the curve: 0.9176
#95% CI: 0.8523-0.983 (DeLong)
roc3=roc(yTrain, as.numeric(pred3[,2]), ci = TRUE)
roc3
#Area under the curve: 0.953
#95% CI: 0.9163-0.9896 (DeLong)
roc4=roc(yTrain, as.numeric(pred4[,2]), ci = TRUE)
roc4
#Area under the curve: 0.9724
#95% CI: 0.9464-0.9985 (DeLong)
roc5=roc(yTrain, as.numeric(pred5[,2]), ci = TRUE)
roc5
#Area under the curve: 0.952
#95% CI: 0.9178-0.9863 (DeLong)
roc6=roc(yTrain, as.numeric(pred6[,2]), ci = TRUE)
roc6
#Area under the curve: 0.9965
#95% CI: 0.9911-1 (DeLong)
roc7=roc(yTrain, as.numeric(pred7[,2]), ci = TRUE)
roc7
#Area under the curve: 0.9507
#95% CI: 0.9141-0.9873 (DeLong)
roc8=roc(yTrain, as.numeric(pred8[,2]), ci = TRUE)
roc8
#Area under the curve: 0.8205
#95% CI: 0.737-0.9041 (DeLong)


pdf(file="结果/ROC.pdf", width=5, height=5)
plot(roc1, print.auc=F, legacy.axes=T, main="", col="chocolate")
plot(roc2, print.auc=F, legacy.axes=T, main="", col="aquamarine3", add=T)
plot(roc3, print.auc=F, legacy.axes=T, main="", col="bisque3", add=T)
plot(roc4, print.auc=F, legacy.axes=T, main="", col="burlywood", add=T)
plot(roc5, print.auc=F, legacy.axes=T, main="", col="darkgoldenrod3", add=T)
plot(roc6, print.auc=F, legacy.axes=T, main="", col="darkolivegreen3", add=T)
plot(roc7, print.auc=F, legacy.axes=T, main="", col="dodgerblue3", add=T)
plot(roc8, print.auc=F, legacy.axes=T, main="", col="darksalmon", add=T)

legend('bottomright',
       c(paste0('RF: ',sprintf("%.03f",roc1$auc)),
         paste0('SVM: ',sprintf("%.03f",roc2$auc)),
         paste0('GLM: ',sprintf("%.03f",roc3$auc)),
         paste0('GBM: ',sprintf("%.03f",roc4$auc)),
         paste0('KNN: ',sprintf("%.03f",roc5$auc)),
         paste0('NNET: ',sprintf("%.03f",roc6$auc)),
         paste0('LASSO: ',sprintf("%.03f",roc7$auc)),
         paste0('DT: ',sprintf("%.03f",roc8$auc))),
       
       
       col=c("chocolate","aquamarine3","bisque3",
             "burlywood","darkgoldenrod3","darkolivegreen3",
             "dodgerblue3","darksalmon"), lwd=2, bty = 'n')
dev.off()
















#对四种方法进行基因的重要性分析,得到四种方法基因重要性评分

##此处运行时间较长，每一种运算方式大约半分钟，耐心等待


importance_rf<-variable_importance(
  explainer_rf,
  loss_function = loss_root_mean_square
)
importance_svm<-variable_importance(
  explainer_svm,
  loss_function = loss_root_mean_square
)
importance_glm<-variable_importance(
  explainer_glm,
  loss_function = loss_root_mean_square
)

importance_knn<-variable_importance(
  explainer_knn,
  loss_function = loss_root_mean_square
)
importance_gbm<-variable_importance(
  explainer_gbm,
  loss_function = loss_root_mean_square
)
importance_nnet<-variable_importance(
  explainer_nnet,
  loss_function = loss_root_mean_square
)
importance_lasso<-variable_importance(
  explainer_lasso,
  loss_function = loss_root_mean_square
)
importance_dt<-variable_importance(
  explainer_dt,
  loss_function = loss_root_mean_square
)

ncol(data)

#绘制基因重要性图形
pdf(file="结果/重要性.pdf", width=15, height=20)
plot(importance_rf[c(1,(ncol(data)-3):(ncol(data)+1)),],
     importance_svm[c(1,(ncol(data)-3):(ncol(data)+1)),],
     importance_gbm[c(1,(ncol(data)-3):(ncol(data)+1)),],
     importance_knn[c(1,(ncol(data)-3):(ncol(data)+1)),],
     importance_nnet[c(1,(ncol(data)-3):(ncol(data)+1)),],
     importance_lasso[c(1,(ncol(data)-3):(ncol(data)+1)),],
     importance_dt[c(1,(ncol(data)-3):(ncol(data)+1)),],
     importance_glm[c(1,(ncol(data)-3):(ncol(data)+1)),])


dev.off()




#输出重要性评分最高的基因
geneNum=5     #设置基因的数目
write.table(importance_rf[(ncol(data)-geneNum+2):(ncol(data)+1),], file="结果/核心特征RF.txt", sep="\t", quote=F, row.names=F)
write.table(importance_svm[(ncol(data)-geneNum+2):(ncol(data)+1),], file="结果/核心特征SVM.txt", sep="\t", quote=F, row.names=F)
write.table(importance_glm[(ncol(data)-geneNum+2):(ncol(data)+1),], file="结果/核心特征GLM.txt", sep="\t", quote=F, row.names=F)
write.table(importance_gbm[(ncol(data)-geneNum+2):(ncol(data)+1),], file="结果/核心特征GBM.txt", sep="\t", quote=F, row.names=F)
write.table(importance_knn[(ncol(data)-geneNum+2):(ncol(data)+1),], file="结果/核心特征KNN.txt", sep="\t", quote=F, row.names=F)
write.table(importance_nnet[(ncol(data)-geneNum+2):(ncol(data)+1),], file="结果/核心特征NNET.txt", sep="\t", quote=F, row.names=F)
write.table(importance_lasso[(ncol(data)-geneNum+2):(ncol(data)+1),], file="结果/核心特征LASSO.txt", sep="\t", quote=F, row.names=F)
write.table(importance_dt[(ncol(data)-geneNum+2):(ncol(data)+1),], file="结果/核心特征DT.txt", sep="\t", quote=F, row.names=F)



