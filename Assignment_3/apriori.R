install.packages("arules", dependencies=TRUE)
library(arules)
txn = read.transactions(file="c:/users/olubisi/desktop/apriori.csv", rm.duplicates= TRUE, format="basket",sep=",");
summary(txn)
inspect(txn[1:5]) 
itemFrequency(txn[,1:5]) 
itemFrequencyPlot(txn, support = 0.1) 
itemFrequencyPlot(txn, topN = 3)
apriori(txn)
groceryrules <- apriori(txn, parameter = list(support =0.6,confidence = 0.70, minlen = 2)) 
groceryrules 
summary(groceryrules)
inspect(groceryrules[1:8])

inspect(sort(groceryrules, by = "lift")[1:8])

if(sessionInfo()['basePkgs']=="tm" | sessionInfo()['otherPkgs']=="tm"){
  detach(package:tm, unload=TRUE)
}

inspect(groceryrules)


#Alternative to inspect() is to convert rules to a dataframe and then use View()
df_basket <- as(groceryrules,"data.frame")
View(df_basket)

library(arulesViz)
plot(groceryrules)
plot(groceryrules, method = "grouped", control = list(k = 5))
plot(groceryrules, method="graph", control=list(type="items"))
#plot(groceryrules, method="paracoord",  control=list(alpha=.5, reorder=TRUE))

