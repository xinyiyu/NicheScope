library(diceR)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  stop("Please provide dataname. Usage: Rscript run.R data_name")
}

dataname <- args[1]
outdir <- args[2]
cat("Running for dataname:", dataname, "\n")

#Import data.
LastStep_OutputFolderName <- sprintf("%s/%s/Step2_Output/", outdir, dataname)
NodeMask <- read.csv(paste0(LastStep_OutputFolderName, "Run1/NodeMask.csv"), header = FALSE)
nonzero_ind <- which(NodeMask$V1 == 1)

#Find the file names of all soft TCN assignment matrices.
allSoftClustFile <- list.files(path = LastStep_OutputFolderName, pattern = "TCN_AssignMatrix1.csv", recursive = TRUE)
allHardClustLabel <- vector()

for (i in 1:length(allSoftClustFile)) {
  
  ClustMatrix <- read.csv(paste0(LastStep_OutputFolderName, allSoftClustFile[i]), header = FALSE, sep = ",")
  ClustMatrix <- ClustMatrix[nonzero_ind,]
  HardClustLabel <- apply(as.matrix(ClustMatrix), 1, which.max)
  rm(ClustMatrix)
  
  allHardClustLabel <- cbind(allHardClustLabel, as.vector(HardClustLabel))
  
} #end of for.

finalClass <- diceR::majority_voting(allHardClustLabel, is.relabelled = FALSE)

ThisStep_OutputFolderName <- sprintf("%s/%s/Step3_Output", outdir, dataname)
if (file.exists(ThisStep_OutputFolderName)){
    unlink(ThisStep_OutputFolderName, recursive=TRUE)  #delete the folder if already exists.
} 
dir.create(ThisStep_OutputFolderName)

write.table(finalClass, file = paste0(ThisStep_OutputFolderName, "/TCNLabel_MajorityVoting.csv"), append = FALSE, quote = FALSE, row.names = FALSE, col.names = FALSE)


