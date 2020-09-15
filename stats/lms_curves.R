#name: lms_curves_commented_for_manuscript_submission_9_4_2020
#statistician: Camden Bay
#date: 9/4/2020
#purpose: fit generalized additive models for reference curves


install.packages("gamlss") #install required package
library(gamlss)

#############################################################################################
#DATA MANAGEMENT#############################################################################
#############################################################################################

data_import <- read.csv("",header=T) #read-in main dataset

muscle_area_data <- data_import[c("MRN","Age","MuscleArea_cm2","Gender","Race")]
subc_area_data <- data_import[c("MRN","Age","SubcArea_cm2","Gender","Race")]
visc_area_data <- data_import[c("MRN","Age","ViscFatArea_cm2","Gender","Race")]
smi_data <- data_import[c("MRN","Age","SMI","Gender","Race")]
sfi_data <- data_import[c("MRN","Age","SFI","Gender","Race")]
vfi_data <- data_import[c("MRN","Age","VFI","Gender","Race")]
weight_data <- data_import[c("MRN","Age","Weight_KG","Gender","Race")]
bmi_data <- data_import[c("MRN","Age","BMI","Gender","Race")]
height_data <- data_import[c("MRN","Age","Height_M","Gender","Race")]

############################################################################################
#MODEL FITTING##############################################################################
############################################################################################

###############################################################################
#MuscleArea_cm2
###############################################################################

##################white, male

data_white_male_ma <- muscle_area_data[which(muscle_area_data$Gender=="Male" & muscle_area_data$Race=="White Not Hispanic"
                                       & muscle_area_data$Age <= 90),] #subsets data by gender, race, and age

lms_white_male_ma <- lms(x=Age,y=MuscleArea_cm2,data=data_white_male_ma, families="BCTo",
                   method.pb="GAIC", cent=c(3,5,10,25,50,75,90,95,97)) #performs model fitting

Q.stats(lms_white_male_ma, xvar=data_white_male_ma$Age) #diagnostics
wp(lms_white_male_ma, xvar=data_white_male_ma$Age, n.inter=4) #diagnostics

##################white, female

data_white_female_ma <- muscle_area_data[which(muscle_area_data$Gender=="Female" & muscle_area_data$Race=="White Not Hispanic"
                                       & muscle_area_data$Age <= 90),]

lms_white_female_ma <- lms(x=Age,y=MuscleArea_cm2,data=data_white_female_ma, families="BCTo",
                   method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_female_ma, xvar=data_white_female_ma$Age)
wp(lms_white_female_ma, xvar=data_white_female_ma$Age, n.inter=6)

##################black, male
data_black_male_ma <- muscle_area_data[which(muscle_area_data$Gender=="Male" & muscle_area_data$Race=="Black" &
                                         muscle_area_data$Age <= 70),]

lms_black_male_ma <- lms(x=Age,y=MuscleArea_cm2,data=data_black_male_ma, families="BCTo",
                   method.pb="GAIC", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_male_ma, xvar=data_black_male_ma$Age)
wp(lms_black_male_ma, xvar=data_black_male_ma$Age, n.inter=6)

##################black, female
data_black_female_ma <- muscle_area_data[which(muscle_area_data$Gender=="Female" & muscle_area_data$Race=="Black" &
                                         muscle_area_data$Age <= 75),]

lms_black_female_ma <- lms(x=Age,y=MuscleArea_cm2,data=data_black_female_ma, families="BCTo",
                   method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_female_ma, xvar=data_black_female_ma$Age)
wp(lms_black_female_ma, xvar=data_black_female_ma$Age, n.inter=6)

####################################################################################
#SubcArea_cm2
####################################################################################

###################white, male
data_white_male_sa <- subc_area_data[which(subc_area_data$Gender=="Male" & subc_area_data$Race=="White Not Hispanic"
                                     & subc_area_data$Age <= 90),]

lms_white_male_sa <- lms(x=Age,y=SubcArea_cm2,data=data_white_male_sa, families="BCTo",
                   method.pb="GAIC", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_male_sa, xvar=data_white_male_sa$Age)
wp(lms_white_male_sa, xvar=data_white_male_sa$Age, n.inter=4)

###################white, female
data_white_female_sa <- subc_area_data[which(subc_area_data$Gender=="Female" & subc_area_data$Race=="White Not Hispanic"
                                     & subc_area_data$Age <= 90),]

lms_white_female_sa <- lms(x=Age,y=SubcArea_cm2,data=data_white_female_sa , families = "BCTo", x.trans=T,mu.df=2,sigma.df=1,
                   method.pb="GAIC", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_female_sa, xvar=data_white_female_sa $Age)
wp(lms_white_female_sa, xvar=data_white_female_sa $Age, n.inter=6)

###################black, male
data_black_male_sa <- subc_area_data[which(subc_area_data$Gender=="Male" & subc_area_data$Race=="Black"
                                     & subc_area_data$Age <= 70),]

lms_black_male_sa <- lms(x=Age,y=SubcArea_cm2,data=data_black_male_sa, mu.df=2,sigma.df=3,
                   method.pb="ML", cent=c(5,10,25,50,75,90,95))

Q.stats(lms_black_male_sa, xvar=data_black_male_sa$Age)
wp(lms_black_male_sa, xvar=data_black_male_sa$Age, n.inter=6)

###################black, female
data_black_female_sa <- subc_area_data[which(subc_area_data$Gender=="Female" & subc_area_data$Race=="Black"
                                     & subc_area_data$Age <= 75),]

lms_black_female_sa <- lms(x=Age,y=SubcArea_cm2,data=data_black_female_sa, sigma.df=2,mu.df=2,
                   method.pb="ML", cent=c(5,10,25,50,75,90,95))

Q.stats(lms_black_female_sa, xvar=data_black_female_sa$Age)
wp(lms_black_female_sa, xvar=data_black_female_sa$Age, n.inter=6)

##############################################################################
#ViscFatArea_cm2 ()
##############################################################################

###############white, male
data_white_male_va <- visc_area_data[which(visc_area_data$Gender=="Male" & visc_area_data$Race=="White Not Hispanic"
                                     & visc_area_data$Age <= 90),]

con = gamlss.control(n.cyc=100)
lms_white_male_va <- lms(x=Age,y=ViscFatArea_cm2,data=data_white_male_va, families="BCTo", x.trans=F, control=con,
                   method.pb="GAIC", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_male_va, xvar=data_white_male_va$Age)
wp(lms_white_male_va, xvar=data_white_male_va$Age, n.inter=6)

################white, female
data_white_female_va <- visc_area_data[which(visc_area_data$Gender=="Female" & visc_area_data$Race=="White Not Hispanic"
                                     & visc_area_data$Age <= 90),]

con = gamlss.control(n.cyc=100)
lms_white_female_va  <- lms(x=Age,y=ViscFatArea_cm2,data=data_white_female_va , families="BCTo", x.trans=T, control=con,
                   method.pb="GAIC", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_female_va, xvar=data_white_female_va $Age)
wp(lms_white_female_va, xvar=data_white_female_va $Age, n.inter=6)

################black, male
data_black_male_va <- visc_area_data[which(visc_area_data$Gender=="Male" & visc_area_data$Race=="Black"
                                     & visc_area_data$Age <= 70),]

con = gamlss.control(n.cyc=100)
lms_black_male_va <- lms(x=Age,y=ViscFatArea_cm2,data=data_black_male_va, families="BCTo", x.trans=T, control=con,
                   method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_male_va, xvar=data_black_male_va$Age)
wp(lms_black_male_va, xvar=data_black_male_va$Age, n.inter=6)

################black, female
data_black_female_va <- visc_area_data[which(visc_area_data$Gender=="Female" & visc_area_data$Race=="Black"
                                     & visc_area_data$Age <= 75),]

con = gamlss.control(n.cyc=100)
lms_black_female_va <- lms(x=Age,y=ViscFatArea_cm2,data=data_black_female_va, families="BCTo", x.trans=T, control=con,
                   method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_female_va, xvar=data_black_female_va$Age)
wp(lms_black_female_va, xvar=data_black_female_va$Age, n.inter=6)

#################################################################################
#SMI
#################################################################################

###################white, male
data_white_male_smi <- smi_data[which(smi_data$Gender=="Male" & smi_data$Race=="White Not Hispanic"
                               & smi_data$Age <= 90),]

lms_white_male_smi <- lms(x=Age,y=SMI,data=data_white_male_smi, families="BCTo",
                   method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_male_smi, xvar=data_white_male_smi$Age)
wp(lms_white_male_smi, xvar=data_white_male_smi$Age, n.inter=4)

###################white, female
data_white_female_smi <- smi_data[which(smi_data$Gender=="Female" & smi_data$Race=="White Not Hispanic"
                               & smi_data$Age <= 90),]

lms_white_female_smi <- lms(x=Age,y=SMI,data=data_white_female_smi, families="BCTo",
                   method.pb="GAIC", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_female_smi, xvar=data_white_female_smi$Age)
wp(lms_white_female_smi, xvar=data_white_female_smi$Age, n.inter=6)

#######################black, male
data_black_male_smi <- smi_data[which(smi_data$Gender=="Male" & smi_data$Race=="Black"
                               & smi_data$Age <= 70),]

lms_black_male_smi <- lms(x=Age,y=SMI,data=data_black_male_smi, families="BCTo", transform=T,
                   method.pb="GAIC", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_male_smi, xvar=data_black_male_smi$Age)
wp(lms_black_male_smi, xvar=data_black_male_smi$Age, n.inter=6)

#######################black, female
data_black_female_smi <- smi_data[which(smi_data$Gender=="Female" & smi_data$Race=="Black" & smi_data$Age <= 75),]

lms_black_female_smi <- lms(x=Age,y=SMI,data=data_black_female_smi, families="BCTo",
                   method.pb="GAIC", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_female_smi, xvar=data_black_female_smi$Age)
wp(lms_black_female_smi, xvar=data_black_female_smi$Age, n.inter=6)

#################################################################################
#SFI
#################################################################################

#####################white, male
data_white_male_sfi <- sfi_data[which(sfi_data$Gender=="Male" & sfi_data$Race=="White Not Hispanic"
                               & sfi_data$Age <= 90),]

lms_white_male_sfi <- lms(x=Age,y=SFI,data=data_white_male_sfi, families = "BCTo",
                   method.pb="GAIC", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_male_sfi, xvar=data_white_male_sfi$Age)
wp(lms_white_male_sfi, xvar=data_white_male_sfi$Age, n.inter=4)

#####################white, female
data_white_female_sfi <- sfi_data[which(sfi_data$Gender=="Female" & sfi_data$Race=="White Not Hispanic"
                               & sfi_data$Age <= 90),]

lms_white_female_sfi <- lms(x=Age,y=SFI,data=data_white_female_sfi, families = "BCTo", x.trans=T, mu.df=3.0, sigma.df=0.5,
                   method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_female_sfi, xvar=data_white_female_sfi$Age)
wp(lms_white_female_sfi, xvar=data_white_female_sfi$Age, n.inter=6)

#####################black, male
data_black_male_sfi <- sfi_data[which(sfi_data$Gender=="Male" & sfi_data$Race=="Black" & sfi_data$Age <= 70),]

con = gamlss.control(n.cyc=100)
lms_black_male_sfi <- lms(x=Age,y=SFI,data=data_black_male_sfi, x.trans=T, tau.df=0.5, mu.df=0.5,nu.df=0.5,sigma.df=0.5,
                          families="BCTo", control=con,
                   method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_male_sfi, xvar=data_black_male_sfi$Age)
wp(lms_black_male_sfi, xvar=data_black_male_sfi$Age, n.inter=6)

######################black, female
data_black_female_sfi <- sfi_data[which(sfi_data$Gender=="Female" & sfi_data$Race=="Black" & sfi_data$Age <= 75),]

con = gamlss.control(n.cyc=1000)
lms_black_female_sfi <- lms(x=Age,y=SFI,data=data_black_female_sfi, control=con, families="BCTo", tau.df=2,
                   method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_female_sfi, xvar=data_black_female_sfi$Age)
wp(lms_black_female_sfi, xvar=data_black_female_sfi$Age, n.inter=6)

#################################################################################
#VFI
#################################################################################

###################white, male
data_white_male_vfi <- vfi_data[which(vfi_data$Gender=="Male" & vfi_data$Race=="White Not Hispanic"
                               & vfi_data$Age < 90),]

con = gamlss.control(n.cyc=100)
lms_white_male_vfi <- lms(x=Age,y=VFI,data=data_white_male_vfi, families="BCTo", x.trans=T, control=con,
                   method.pb="GAIC", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_male_vfi, xvar=data_white_male_vfi$Age)
wp(lms_white_male_vfi, xvar=data_white_male_vfi$Age, n.inter=6)

###################white, female
data_white_female_vfi <- vfi_data[which(vfi_data$Gender=="Female" & vfi_data$Race=="White Not Hispanic"
                               & vfi_data$Age < 90),]

con = gamlss.control(n.cyc=100)
lms_white_female_vfi <- lms(x=Age,y=VFI,data=data_white_female_vfi, families="BCTo", x.trans=T, control=con,
                   method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_female_vfi, xvar=data_white_female_vfi$Age)
wp(lms_white_female_vfi, xvar=data_white_female_vfi$Age, n.inter=6)

##########################black, male
data_black_male_vfi <- vfi_data[which(vfi_data$Gender=="Male" & vfi_data$Race=="Black"
                               & vfi_data$Age <= 70),]

con = gamlss.control(n.cyc=1000)
lms_black_male_vfi <- lms(x=Age,y=VFI,data=data_black_male_vfi, x.trans=T, control=con,families="BCTo",
                          mu.df=2.5, sigma.df=2.5, tau.df=2.5, nu.df=2.5,
                   method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_male_vfi, xvar=data_black_male_vfi$Age)
wp(lms_black_male_vfi, xvar=data_black_male_vfi$Age, n.inter=6)

#########################black, female
data_black_female_vfi <- vfi_data[which(vfi_data$Gender=="Female" & vfi_data$Race=="Black"
                               & vfi_data$Age <= 75),]

con = gamlss.control(n.cyc=100)
lms_black_female_vfi <- lms(x=Age,y=VFI,data=data_black_female_vfi, families="BCTo", x.trans=T, control=con,
                   method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_female_vfi, xvar=data_black_female_vfi$Age)
wp(lms_black_female_vfi, xvar=data_black_female_vfi$Age, n.inter=6)


###############################################################################
#BMI
###############################################################################

##################white, male

data_white_male_bmi <- bmi_data[which(bmi_data$Gender=="Male" & bmi_data$Race=="White Not Hispanic"
                                      & bmi_data$Age <= 90),]

lms_white_male_bmi <- lms(x=Age,y=BMI,data=data_white_male_bmi, families="BCTo",
                          method.pb="GAIC", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_male_bmi, xvar=data_white_male_bmi$Age)
wp(lms_white_male_bmi, xvar=data_white_male_bmi$Age, n.inter=4)

##################white, female

data_white_female_bmi <- bmi_data[which(bmi_data$Gender=="Female" & bmi_data$Race=="White Not Hispanic"
                                        & bmi_data$Age <= 90),]

lms_white_female_bmi <- lms(x=Age,y=BMI,data=data_white_female_bmi, families="BCTo",
                            method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_female_bmi, xvar=data_white_female_bmi$Age)
wp(lms_white_female_bmi, xvar=data_white_female_bmi$Age, n.inter=6)

##################black, male
data_black_male_bmi <- bmi_data[which(bmi_data$Gender=="Male" & bmi_data$Race=="Black" &
                                        bmi_data$Age <= 70),]

con = gamlss.control(n.cyc=100)
lms_black_male_bmi <- lms(x=Age,y=BMI,data=data_black_male_bmi, control=con, x.trans=T, families = "BCTo",
                          method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_male_bmi, xvar=data_black_male_bmi$Age)
wp(lms_black_male_bmi, xvar=data_black_male_bmi$Age, n.inter=6)

##################black, female
data_black_female_bmi <- bmi_data[which(bmi_data$Gender=="Female" & bmi_data$Race=="Black" &
                                          bmi_data$Age <= 75),]

lms_black_female_bmi <- lms(x=Age,y=BMI,data=data_black_female_bmi, families="BCTo",
                            method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_female_bmi, xvar=data_black_female_bmi$Age)
wp(lms_black_female_bmi, xvar=data_black_female_bmi$Age, n.inter=6)


#################################################################################
#body weight
#################################################################################

##################white, male

data_white_male_weight <- weight_data[which(weight_data$Gender=="Male" & weight_data$Race=="White Not Hispanic"
                                             & weight_data$Age <= 90),]

lms_white_male_weight <- lms(x=Age,y=Weight_KG,data=data_white_male_weight, families="BCTo",
                         method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_male_weight, xvar=data_white_male_weight$Age)
wp(lms_white_male_weight, xvar=data_white_male_weight$Age, n.inter=4)

##################white, female

data_white_female_weight <- weight_data[which(weight_data$Gender=="Female" & weight_data$Race=="White Not Hispanic"
                                               & weight_data$Age <= 90),]

con = gamlss.control(n.cyc=100)
lms_white_female_weight <- lms(x=Age,y=Weight_KG,data=data_white_female_weight,control=con, x.tran=T,
                           families="BCTo", method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_female_weight, xvar=data_white_female_weight$Age)
wp(lms_white_female_weight, xvar=data_white_female_weight$Age, n.inter=6)

##################black, male
data_black_male_weight <- weight_data[which(weight_data$Gender=="Male" & weight_data$Race=="Black" &
                                              weight_data$Age <= 70),]

lms_black_male_weight <- lms(x=Age,y=Weight_KG,data=data_black_male_weight, families="BCPEo",
                         method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_male_weight, xvar=data_black_male_weight$Age)
wp(lms_black_male_weight, xvar=data_black_male_weight$Age, n.inter=6)

##################black, female
data_black_female_weight <- weight_data[which(weight_data$Gender=="Female" & weight_data$Race=="Black" &
                                                weight_data$Age <= 75),]

lms_black_female_weight <- lms(x=Age,y=Weight_KG,data=data_black_female_weight, x.trans=T, families="BCTo",
                           method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_female_weight, xvar=data_black_female_weight$Age)
wp(lms_black_female_weight, xvar=data_black_female_weight$Age, n.inter=6)


#################################################################################
#body height
#################################################################################

##################white, male

data_white_male_height <- height_data[which(height_data$Gender=="Male" & height_data$Race=="White Not Hispanic"
                                            & height_data$Age <= 90),]

lms_white_male_height <- lms(x=Age,y=Height_M,data=data_white_male_height,
                             method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_male_height, xvar=data_white_male_height$Age)
wp(lms_white_male_height, xvar=data_white_male_height$Age, n.inter=4)

##################white, female

data_white_female_height <- height_data[which(height_data$Gender=="Female" & height_data$Race=="White Not Hispanic"
                                              & height_data$Age <= 90),]

con = gamlss.control(n.cyc=100)
lms_white_female_height <- lms(x=Age,y=Height_M,data=data_white_female_height,control=con, x.tran=T,
                               method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_white_female_height, xvar=data_white_female_height$Age)
wp(lms_white_female_height, xvar=data_white_female_height$Age, n.inter=6)

##################black, male
data_black_male_height <- height_data[which(height_data$Gender=="Male" & height_data$Race=="Black" &
                                              height_data$Age <= 70),]

lms_black_male_height <- lms(x=Age,y=Height_M,data=data_black_male_height,x.tran=T,families="BCPEo",
                             method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_male_height, xvar=data_black_male_height$Age)
wp(lms_black_male_height, xvar=data_black_male_height$Age, n.inter=6)

##################black, female
data_black_female_height <- height_data[which(height_data$Gender=="Female" & height_data$Race=="Black" &
                                                height_data$Age <= 75),]

lms_black_female_height <- lms(x=Age,y=Height_M,data=data_black_female_height, x.trans=T,
                               method.pb="ML", cent=c(3,5,10,25,50,75,90,95,97))

Q.stats(lms_black_female_height, xvar=data_black_female_height$Age)
wp(lms_black_female_height, xvar=data_black_female_height$Age, n.inter=6)


#Model objects have now been constructed for all response, gender, and race combinations. They should be saved as
#permanent objects. They can be used with the centiles() function to create reference curves and the z.scores()
#function to generate z-scores.
