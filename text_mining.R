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
library(Rtsne)
library(lsa)
#preparing datasets
spotify <- read_csv("spotify.csv")
billboard <- read_csv("billboard.csv")
#spotify
spotify<-spotify%>% filter(language=='en') #filtering for language
spotify<-spotify%>% filter(year(as.Date(track_album_release_date))>1998) #filtering dates
spotify<-spotify%>% filter(year(as.Date(track_album_release_date))<2020) #filtering dates
spotify<-spotify %>% distinct(track_name, track_artist, .keep_all = TRUE) #removing duplicates
spotify<-spotify %>% distinct(lyrics, .keep_all = TRUE) #removing duplicates
spotify<- subset(spotify, select = c(track_name, track_artist,lyrics,
                                     track_popularity, track_album_release_date,
                                     playlist_genre,energy,valence)) #removing unnecessary columns
#---putting labels
ggplot(data=spotify, aes(x=valence,y=energy))+geom_point(alpha = 1/10)+theme_minimal()+
  geom_hline(yintercept=0.5,linetype="dashed")+geom_vline(xintercept=0.5,linetype="dashed")
quantile(spotify$energy, probs = c(0.25,0.75))
quantile(spotify$valence, probs = c(0.25,0.75)) 
spotify<-spotify %>%
  mutate(label = case_when(
    (energy>0.837 & valence>0.649) ~ "Happy",
    (energy>0.837 & valence<0.312) ~ "Angry",
    (energy<0.5 & valence>0.649) ~ "Relaxed", #1st quartile is 0.572, so we will use 0.5
    (energy<0.5 & valence<0.312) ~ "Sad"    #1st quartile is 0.572, so we will use 0.5
  ))

#---counting how many songs in each mood category
spotify<-na.omit(spotify)
spotify %>% count(label)
spotify<-spotify%>%filter(label!="Relaxed")

#billboard

#---removing unnecessary data
billboard<- subset(billboard, select = c(Artists, Name,Peak.position, Weeks.on.chart,Week,Lyrics))
billboard <- na.omit(billboard)
billboard<-billboard%>%filter(Peak.position==1)
billboard<-billboard %>% distinct(Name, .keep_all = TRUE)

#data clean-up
spotify$lyrics <- str_replace_all(as.character(spotify$lyrics), '[^a-zA-Z \']',' ') #replacing all special characters
spotify <- spotify %>% mutate(lyrics_clean = lyrics %>%
                                str_to_lower() %>% #changing all to lower case 
                                replace_contraction() %>%#replacing contractions 
                                replace_word_elongation() %>% #replacing elongations 
                                str_squish() %>% str_trim()) #removing whitespace

#checking for most frequent words in each category
#---Happy
happy<-spotify %>% filter(label=="Happy")
corp <- VCorpus(VectorSource(happy$lyrics_clean)) #word corpus
tdm <- corp %>% 
  tm_map(removeWords, stopwords("en")) %>% #removing stopwords
  tm_map(stemDocument) %>% #stemming
  TermDocumentMatrix() #matix

m <- as.matrix(tdm)
v <- sort(rowSums(m),decreasing=TRUE)
d_happy <- data.frame(word = names(v),freq=v)

#---Sad
sad<-spotify %>% filter(label=="Sad")
corp <- VCorpus(VectorSource(sad$lyrics_clean)) #word corpus
tdm <- corp %>% 
  tm_map(removeWords, stopwords("en")) %>% #removing stopwords
  tm_map(stemDocument) %>% #stemming
  TermDocumentMatrix() #matix

m <- as.matrix(tdm)
v <- sort(rowSums(m),decreasing=TRUE)
d_sad <- data.frame(word = names(v),freq=v)

#---Angry
angry<-spotify %>% filter(label=="Angry")
corp <- VCorpus(VectorSource(angry$lyrics_clean)) #word corpus
tdm <- corp %>% 
  tm_map(removeWords, stopwords("en")) %>% #removing stopwords
  tm_map(stemDocument) %>% #stemming
  TermDocumentMatrix() #matix

m <- as.matrix(tdm)
v <- sort(rowSums(m),decreasing=TRUE)
d_angry <- data.frame(word = names(v),freq=v)

#developing TF-IDF (Term Frequency - Inverse Document) Martix
stop<-c('love','will','like','can','know','now','just') #optional - adding more stopwords
NLP_tokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), 1:3), paste, collapse = "_"), use.names = FALSE)
} #function used to create dtm
control_list_ngram = list(tokenize = NLP_tokenizer,
                          weighting = weightTfIdf) #function used to create dtm
corp <- VCorpus(VectorSource(spotify$lyrics_clean)) #word corpus
big_tfidf <- corp %>% 
  tm_map(removeWords, c(stopwords("en"),stop)) %>% #removing stopwords
  tm_map(stemDocument) %>% #stemming
  DocumentTermMatrix(control_list_ngram) #adding TF-IDF weights and tokenizing

#checking which words have the highest tf-idf weights
m <- as.matrix(big_tfidf)
v <- sort(colSums(m),decreasing=TRUE)
d <- data.frame(tf=v)

#DTM has too many variables for models to run

#1 feature selection --- filtering by tf-idf score
tfidf<-filter_words(big_tfidf, 
                    min=quantile(d$tf, 0.995) #different values of min generate different results
                    ) #so its good to experiment with them - e.g. min=1 gives good results

#MODELING
#---changing matrix to data frame
dat.clean <- as.data.frame(as.matrix(tfidf), stringsAsFactors = F)
#---adding label
new.dat <- cbind(dat.clean, data.frame(labelY = as.factor(spotify$label)))
#---adding colnames
colnames(new.dat) <- make.names(colnames(new.dat))
#---spliting into training and testing dataset
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