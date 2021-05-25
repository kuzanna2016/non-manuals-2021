install.packages("blme")
install.packages("optimx")
install.packages('multcomp')
library(blme) #bayesian model with covert prior for covariance matrix
library(optimx) #optimizer
library("car")
library('multcomp')

# load data
pose_Rx <- read.csv("pose_Rx_fpca_scores.csv")
inner <- read.csv("inner_perp_dist39_42_3d_pred_mlp_8_diff_fpca_scores.csv")
outer <- read.csv("outer_perp_dist39_42_3d_pred_mlp_8_diff_fpca_scores.csv")

# inner brows
inner$sType <- as.factor(inner$sType)
inner$deaf<-as.factor(inner$deaf)

contrast <- cbind (c(1/2, -1/2)) #deaf, hearing
colnames (contrast) <- c("+deaf")
contrasts (inner$deaf) <- contrast
contrasts(inner$deaf)

# for PC_1
modelIntbPC1<-blmer(PC_1 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=inner, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC1)
Anova(modelIntbPC1)

innerPC1 <- glht(modelIntbPC1, linfct = mcp(sType = "Tukey"))

# here and onward you can choose test like this, but we stick to the default one
# summary(innerPC1, test = adjusted(type = "bonferroni"))

summary(innerPC1)

# you can plot the results like this
innerPC1.cld <- cld(innerPC1)
plot(innerPC1.cld)

# for PC_2

modelIntbPC2<-blmer(PC_2 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=inner, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC2)
Anova(modelIntbPC2)

innerPC2 <- glht(modelIntbPC2, linfct = mcp(sType = "Tukey"))
summary(innerPC2)

# for PC_3

modelIntbPC3<-blmer(PC_3 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=inner, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC3)
Anova(modelIntbPC3)

innerPC3 <- glht(modelIntbPC3, linfct = mcp(sType = "Tukey"))
summary(innerPC3)

# for PC_4

modelIntbPC4<-blmer(PC_4 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=inner, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC4)
Anova(modelIntbPC4)

innerPC4 <- glht(modelIntbPC4, linfct = mcp(sType = "Tukey"))
summary(innerPC4)

# for PC_5

modelIntbPC5<-blmer(PC_5 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=inner, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC5)
Anova(modelIntbPC5)

innerPC5 <- glht(modelIntbPC5, linfct = mcp(sType = "Tukey"))
summary(innerPC5)


# ============================================================================
# outer brows
outer$sType <- as.factor(outer$sType)
outer$deaf<-as.factor(outer$deaf)

contrast <- cbind (c(1/2, -1/2)) #deaf, hearing
colnames (contrast) <- c("+deaf")
contrasts (outer$deaf) <- contrast
contrasts(outer$deaf)

# for PC_1
modelIntbPC1<-blmer(PC_1 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=outer, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC1)
Anova(modelIntbPC1)

outerPC1 <- glht(modelIntbPC1, linfct = mcp(sType = "Tukey"))
summary(outerPC1)

# for PC_2

modelIntbPC2<-blmer(PC_2 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=outer, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC2)
Anova(modelIntbPC2)

outerPC2 <- glht(modelIntbPC2, linfct = mcp(sType = "Tukey"))
summary(outerPC2)

# for PC_3

modelIntbPC3<-blmer(PC_3 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=outer, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC3)
Anova(modelIntbPC3)

outerPC3 <- glht(modelIntbPC3, linfct = mcp(sType = "Tukey"))
summary(outerPC3)

# for PC_4

modelIntbPC4<-blmer(PC_4 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=outer, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC4)
Anova(modelIntbPC4)

outerPC4 <- glht(modelIntbPC4, linfct = mcp(sType = "Tukey"))
summary(outerPC4)

# for PC_5

modelIntbPC5<-blmer(PC_5 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=outer, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC5)
Anova(modelIntbPC5)

outerPC5 <- glht(modelIntbPC5, linfct = mcp(sType = "Tukey"))
summary(outerPC5)


# ========================================================================

# pose
pose_Rx$sType <- as.factor(pose_Rx$sType)
pose_Rx$deaf<-as.factor(pose_Rx$deaf)

contrast <- cbind (c(1/2, -1/2)) #deaf, hearing
colnames (contrast) <- c("+deaf")
contrasts (pose_Rx$deaf) <- contrast
contrasts(pose_Rx$deaf)

# for PC_1
modelIntbPC1<-blmer(PC_1 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=pose_Rx, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC1)
Anova(modelIntbPC1)

posePC1 <- glht(modelIntbPC1, linfct = mcp(sType = "Tukey"))
summary(posePC1)

# for PC_2

modelIntbPC2<-blmer(PC_2 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=pose_Rx, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC2)
Anova(modelIntbPC2)

posePC2 <- glht(modelIntbPC2, linfct = mcp(sType = "Tukey"))
summary(posePC2)

# for PC_3

modelIntbPC3<-blmer(PC_3 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=pose_Rx, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC3)
Anova(modelIntbPC3)

posePC3 <- glht(modelIntbPC3, linfct = mcp(sType = "Tukey"))
summary(posePC3)

# for PC_4

modelIntbPC4<-blmer(PC_4 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=pose_Rx, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC4)
Anova(modelIntbPC4)

posePC4 <- glht(modelIntbPC4, linfct = mcp(sType = "Tukey"))
summary(posePC4)

# for PC_5

modelIntbPC5<-blmer(PC_5 ~ sType * deaf + (sType|speaker_id) + (sType * deaf|sentence), data=pose_Rx, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb'))) #
summary(modelIntbPC5)
Anova(modelIntbPC5)

posePC5 <- glht(modelIntbPC5, linfct = mcp(sType = "Tukey"))
summary(posePC5)
