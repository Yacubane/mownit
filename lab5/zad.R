setwd("/home/jakubik/Studia/mownit/lab5/")
results = read.csv("matrix_multiplication_O0.csv")

avg_results = aggregate( cbind(naive,better,blas) ~ size, data=results, FUN=mean)
avg_results$naive_sd = aggregate(naive ~ size, data=results, FUN=sd)$naive
avg_results$better_sd = aggregate(better ~ size, data=results, FUN=sd)$better
avg_results$blas_sd = aggregate(blas ~ size, data=results, FUN=sd)$blas

library("ggplot2")

fit = lm(naive ~ poly(as.vector(avg_results[["size"]]), 3, raw=TRUE), data=avg_results)
newdata = data.frame(x = avg_results[["size"]])
newdata$y = predict(fit, newdata)

fit2 = lm(better ~ poly(as.vector(avg_results[["size"]]), 3, raw=TRUE), data=avg_results)
newdata2 = data.frame(x = avg_results[["size"]])
newdata2$y = predict(fit2, newdata2)

fit3 = lm(blas ~ poly(as.vector(avg_results[["size"]]), 3, raw=TRUE), data=avg_results)
newdata3 = data.frame(x = avg_results[["size"]])
newdata3$y = predict(fit3, newdata3)

ggplot() +
geom_point(data = avg_results, aes(size,naive)) +
geom_point(data = avg_results, aes(size,better)) + 
geom_point(data = avg_results, aes(size,blas)) +
geom_errorbar(
  data = avg_results,
  aes(size, naive, ymin = naive - naive_sd, ymax = naive + naive_sd),
  colour = 'red',
  width = 0.4
) +
geom_errorbar(
  data = avg_results,
  aes(size, better, ymin = better - better_sd, ymax = better + better_sd),
  colour = 'red',
  width = 0.4
) +
geom_errorbar(
  data = avg_results,
  aes(size, blas, ymin = blas - blas_sd, ymax = blas + blas_sd),
  colour = 'red',
  width = 0.4
) +
geom_line(data=newdata, aes(x,y)) +
geom_line(data=newdata2, aes(x,y)) +
geom_line(data=newdata3, aes(x,y))

