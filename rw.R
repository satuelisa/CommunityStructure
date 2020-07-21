d = read.csv('rw.csv', header=FALSE, sep=" ", stringsAsFactors=FALSE)
names(d)[1] = "vertex"
names(d)[2] = "density"
names(d)[3] = "label"
names(d)[4] = "degree"
names(d)[5] = "kind"
names(d)[6] = "depth"
names(d)[7] = "measure"
d$kind = as.factor(d$kind)
d$label = as.factor(d$label)
d = d[order(d$vertex),]
density = d$density[1]
library(ggplot2)
p = ggplot(d, aes(x=kind, y=measure, fill=kind)) + geom_violin(trim=TRUE) + scale_y_log10()
p + stat_summary(fun.y=mean, geom="point", shape=21, size=3)
ggsave('rw.png', width=15, height=10, units="cm")
n = dim(d)[1]
d$assigned = rep("None", n)
classes = c("Uniform", "Hierarchical", "Flat")
cuts = numeric()
for (cl in classes) {
    cat(c(cl, summary(d[d$kind == cl,]$measure)), "\n")
    cuts = c(cuts, median(d[d$kind == cl,]$measure))
}
for (v in 1:n) {
    sel = which.min(abs(cuts - rep(d$measure[v], 3)))
    d$assigned[v] = classes[sel]
}
d$correct = d$kind == d$assigned
print(sum(d$correct) / n)
precision = matrix(rep(0, 9), nrow = 3)
r = 1
for (class in classes) {
    s = d[d$kind == class,]
    c = 1
    for (other in classes) {
        t = s[s$assigned == other,]
        precision[r, c] = dim(t)[1]
        c = c + 1
    }
    r = r + 1
}
print(precision)
