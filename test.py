import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import pandas as pd

class Category:
    BOOKS = "BOOKS"
    CLOTHING = "CLOTHING"

train_x =["i love the book book" , "this is a great book", "the fit is great", "i love the shoes"]
train_y =[Category.BOOKS, Category.BOOKS, Category.CLOTHING,Category.CLOTHING]
vectorizer = CountVectorizer(binary= True)
train_x_vectors = vectorizer.fit_transform(train_x)

#vectors = vectorizer.fit_transform(train_x)
# vocab = vectorizer.get_feature_names_out()
# print(vocab)
# print(vectors[0])
# print(vectors[1])

# df = pd.DataFrame(vectors.toarray(), columns = vectorizer.get_feature_names_out())
# print(df)
print(vectorizer.get_feature_names_out())
print(train_x_vectors.toarray())

#using SKV for linear classification
clf_svm = svm.SVC(kernel = 'linear')
clf_svm.fit(train_x_vectors, train_y)

user_input = input("Enter a sentence to classifcy: ")
test_x = vectorizer.transform([user_input])
print(clf_svm.predict(test_x))

#wrod vectors 
