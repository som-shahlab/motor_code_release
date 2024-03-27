library(survival)
library(randomForestSRC)
library(ggplot2)
library(dplyr)
library(ggfortify)
library(reticulate)
library(Matrix)
library(optparse)
library(glmnet)
library(doParallel)

options(rf.cores=16, mc.cores=16)
registerDoParallel(16)

option_list = list(
  make_option("--path", type="character"),
  make_option("--nodesize", type="integer"),
  make_option("--outpath", type="character"),
  make_option("--models", type="character"),
  make_option("--num_bin", type="integer")
);
 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

np = import("numpy")
py = import_builtins()

path = opt$path

time = np$load(paste(path, "deltas.npy", sep="/"))
status = np$load(paste(path, "is_event.npy", sep="/"))
dense = np$load(paste(path, "features.npy", sep="/"))

train_indices = np$load(paste(path, "train_indices.npy", sep="/")) + 1
val_indicies = np$load(paste(path, "val_indices.npy", sep="/")) + 1
test_indicies = np$load(paste(path, "test_indices.npy", sep="/")) + 1

print(length(train_indices))

train_indices = train_indices[time[train_indices] > 0]

print(length(train_indices))

dense_matrix = dense

dense = data.frame(dense, t=time, s=status)

if (opt$models == "survival") {
    r = rfsrc(Surv(t, s) ~ ., dense[train_indices,], verbose=TRUE, ntime=opt$num_bin, nodesize=opt$nodesize)
    results = predict(r, dense)
} else if (opt$models == "cox") {   
    r = cv.glmnet(dense_matrix[train_indices,], Surv(time[train_indices], status[train_indices]), family = "cox", parrallel=TRUE)
    results = predict(r, dense_matrix)
}
    
save(r, file=paste(opt$outpath, "r.robj", sep="/"))


if (opt$models == "cox") {
    np$save(paste(opt$outpath, "hazards.npy", sep="/"), results)
} else if (opt$models == "survival") {
    np$save(paste(opt$outpath, "times.npy", sep="/"), results$time.interest)
    np$save(paste(opt$outpath, "survival.npy", sep="/"), results$survival)
}

file.create(paste(opt$outpath, "done", sep="/"))
