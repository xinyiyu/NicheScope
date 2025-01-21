sparkx_sk <- function(counts,infomat,mc_cores=1){
    Xinfomat 		<- apply(infomat,2,scale,scale=FALSE)
    
    # loc_inv 		<- solve(t(infomat) %*% infomat)
    loc_inv 		<- solve(crossprod(Xinfomat,Xinfomat))
    
    kmat_first 		<- Xinfomat %*% loc_inv
    
    LocDim 			<- ncol(infomat)
    Klam 			<- eigen(crossprod(Xinfomat,kmat_first), only.values=T)$values
    EHL 				<- counts%*%Xinfomat
    numCell 			<- nrow(Xinfomat)
    
    adjust_nominator <- as.vector(rowSums(counts^2,TRUE))
    vec_stat 		 <- apply(EHL,1,function(x){x%*%loc_inv%*%as.matrix(x)})*numCell/adjust_nominator
    
    vec_ybar  		<- as.vector(rowMeans(counts,TRUE))
    vec_ylam 		<- unlist(parallel::mclapply(1:nrow(counts),function(x){1-numCell*vec_ybar[x]^2/adjust_nominator[x]},mc.cores=mc_cores))
    vec_daviesp 	<- unlist(parallel::mclapply(1:nrow(counts),function(x){SPARK::sparkx_pval(x,vec_ylam,Klam,vec_stat)},mc.cores=mc_cores))
    res_sparkx 		<- as.data.frame(cbind(vec_stat,vec_daviesp))
    colnames(res_sparkx) 	<- c("stat","pval")
    gene_ids <- which((res_sparkx$pval>0)&(res_sparkx$pval<1))
    res_sparkx <- res_sparkx[(res_sparkx$pval>0)&(res_sparkx$pval<1),]
    
    allstat <- res_sparkx$stat
    allpvals <- res_sparkx$pval
    comb_pval 	<- sapply(allpvals, SPARK::ACAT)
    pBY 		<- p.adjust(comb_pval,method="BY")
    
    joint_pval 	<- cbind.data.frame(combinedPval=comb_pval,adjustedPval=pBY)
    
    res_sparkx 	<- list(gene_ids=gene_ids,stats=allstat,res_stest=allpvals,res_mtest=joint_pval)
    return(res_sparkx)
}
