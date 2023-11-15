#libraries
library(tidyverse)
library(rminer)
library(dplyr)
library(ggplot2)
library(stringr)
library(textclean)
library(tm)
library(SnowballC)
library(textmineR)
library(e1071)
library(parsnip)
library(rsample)
library(rpart.plot)
library(partykit)
library(tidytext)
library(gofastr)
library(caret)
#preparing datasets
spotify<-spotify%>% filter(language=='en') #filtering for language
spotify<-spotify%>% filter(year(as.Date(track_album_release_date))>1998) #filtering dates
spotify<-spotify%>% filter(year(as.Date(track_album_release_date))<2020) #filtering dates
spotify<-spotify %>% distinct(track_name, track_artist, .keep_all = TRUE) #removing duplicates
spotify<- subset(spotify, select = c(track_name, track_artist,lyrics,
                                     track_popularity, track_album_release_date,
                                     playlist_genre,energy,valence)) #removing unnecessary columns
#putting labels
spotify<-spotify %>%
  mutate(label = case_when(
    (energy>=0.5 & valence>=0.5) ~ "Happy",
    (energy>0.5 & valence<0.5) ~ "Angry",
    (energy<=0.5 & valence>=0.5) ~ "Relaxed",
    (energy<=0.5 & valence<=0.5) ~ "Sad"
  ))
#counting how many songs in each mood category
spotify %>% count(label)
#removing unnecessary columns
billboard<- subset(billboard, select = c(Artists, Name,Peak.position, Weeks.on.chart,Week,Lyrics))
#removing rows with missing values
billboard <- na.omit(billboard) 
#adding a column with overall peak position
setDT(billboard)[, Peak:= min(Peak.position), Name]
#adding a column with overall weeks on chart
setDT(billboard)[, Weeks:= max(Weeks.on.chart), Name]
#changing column name from Weeks to Date
colnames(billboard)[5] ="Date"
#removing duplicates
billboard<-billboard %>% distinct(Name, .keep_all = TRUE)
#removing unnecessary columns
billboard<- subset(billboard, select = c(Artists, Name,Peak, Weeks,Date,Lyrics))
#calculating popularity index
billboard<-billboard %>% mutate(Popularity=(1/Peak+Weeks/max(Weeks))/2*100)
#rounding popularity index
billboard$Popularity<-round(billboard$Popularity)
#data clean-up
#replacing all special characters
spotify$lyrics <- str_replace_all(as.character(spotify$lyrics), '[^a-zA-Z \']',' ')
spotify <- spotify %>% mutate(lyrics_clean = lyrics %>%
                                str_to_lower() %>% #changing all to lower case 
                                replace_contraction() %>%#replacing contractions 
                                replace_word_elongation() %>% #replacing elongations 
                                str_squish() %>% str_trim()) #removing whitespace
#checking for most frequent words in each category
#Happy
happy<-spotify %>% filter(label=="Happy")
corp <- VCorpus(VectorSource(happy$lyrics_clean)) #word corpus
tdm <- corp %>% 
  tm_map(removeWords, stopwords("en")) %>% #removing stopwords
  tm_map(stemDocument) %>% #stemming
  TermDocumentMatrix() #matix

m <- as.matrix(tdm)
v <- sort(rowSums(m),decreasing=TRUE)
d_happy <- data.frame(word = names(v),freq=v)
#Sad
sad<-spotify %>% filter(label=="Sad")
corp <- VCorpus(VectorSource(sad$lyrics_clean)) #word corpus
tdm <- corp %>% 
  tm_map(removeWords, stopwords("en")) %>% #removing stopwords
  tm_map(stemDocument) %>% #stemming
  TermDocumentMatrix() #matix

m <- as.matrix(tdm)
v <- sort(rowSums(m),decreasing=TRUE)
d_sad <- data.frame(word = names(v),freq=v)

#Angry
angry<-spotify %>% filter(label=="Angry")
corp <- VCorpus(VectorSource(angry$lyrics_clean)) #word corpus
tdm <- corp %>% 
  tm_map(removeWords, stopwords("en")) %>% #removing stopwords
  tm_map(stemDocument) %>% #stemming
  TermDocumentMatrix() #matix

m <- as.matrix(tdm)
v <- sort(rowSums(m),decreasing=TRUE)
d_angry <- data.frame(word = names(v),freq=v)

#Relaxed
relaxed<-spotify %>% filter(label=="Relaxed")
corp <- VCorpus(VectorSource(relaxed$lyrics_clean)) #word corpus
tdm <- corp %>% 
  tm_map(removeWords, stopwords("en")) %>% #removing stopwords
  tm_map(stemDocument) %>% #stemming
  TermDocumentMatrix() #matix

m <- as.matrix(tdm)
v <- sort(rowSums(m),decreasing=TRUE)
d_relaxed <- data.frame(word = names(v),freq=v)

#developing TF-IDF (Term Frequency - Inverse Document) Martix
corp <- VCorpus(VectorSource(spotify$lyrics_clean)) #word corpus
corp_dtm <- corp %>% 
  tm_map(removeWords, stopwords("en")) %>% #removing stopwords
  tm_map(stemDocument) %>% #stemming
  DocumentTermMatrix(control = list(weighting = weightTfIdf)) #adding TF-IDF weights

corp <- VCorpus(VectorSource(spotify$lyrics_clean)) #word corpus
corp_dtm <- corp %>% 
  tm_map(removeWords, stopwords("en")) %>% #removing stopwords
  tm_map(stemDocument) %>% #stemming
  TermDocumentMatrix(control = list(weighting = weightTfIdf)) #adding TF-IDF weights

m <- as.matrix(corp_dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)

#DTM has too many variables for models to run, here is my fix from gofastr library 
small_corp_dtm<-filter_words(corp_dtm)
tfidf<-small_corp_dtm
#MODELING
#preparing dataset
#changing to data frame
dat.clean <- as.data.frame(as.matrix(tfidf), stringsAsFactors = F)
#adding label
new.dat <- cbind(dat.clean, data.frame(labelY = as.factor(spotify$label)))
#adding colnames
colnames(new.dat) <- make.names(colnames(new.dat))
#spliting into training and testing dataset
splitter <- initial_split(new.dat, prop = 0.75, strata = "labelY")
train <- training(splitter)
test <- testing(splitter)
#Naive Bayers
NB_model <- naiveBayes(labelY~., data=train)
NB_model_pred<-predict(NB_model, test)
confusionMatrix(data = NB_model_pred, test$labelY)
#Decision Tree
DT_model <- train(labelY~., data = train, method = "rpart2")
DT_model_pred <- predict(DT_model, test)
confusionMatrix(data = DT_model_pred, test$labelY)
#Random Forrest
RF_model <- train(labelY~., data = train, method = "rf")
RF_model_pred <- predict(RF_model, test)
confusionMatrix(data = RF_model_pred, test$labelY)