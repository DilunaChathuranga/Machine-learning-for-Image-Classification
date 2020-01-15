#!/usr/bin/env python
# coding: utf-8

#  Iris Flower dataset

#  Objective: Classify a new flower as belonging to one of the 3 classes.
# 

# In[ ]:


#H.K.D.C.Jayalath-IT16001480
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image
import pydotplus
from sklearn.datasets import load_svmlight_file
import seaborn as sns; sns.set() 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
from sklearn import metrics 
from sklearn.tree import export_graphviz 
import seaborn as sns
from IPython.display import Image


# In[ ]:


#Picture of the three different Iris Flower types:


# In[ ]:


# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[ ]:


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[ ]:


# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# In[ ]:


# Importing the dataset
dataset = pd.read_csv('Iris.csv')


# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


# Remove Id Column
dataset.drop("Id", axis=1, inplace = True)


# In[ ]:


dataset["Species"].value_counts()


# In[ ]:


#load data
iris=datasets.load_iris()
x=iris.data
y=iris.target


# In[ ]:


iris.feature_names
iris.target_names


# In[ ]:


#create decision tree classifier
clf=DecisionTreeClassifier(random_state=0)

#train the modle
model=clf.fit(x,y)


# In[ ]:


dot_data = tree.export_graphviz(model,out_file=None,
                              feature_names=iris.feature_names,
                              class_names=iris.target_names)
#draw graph
graph=pydotplus.graph_from_dot_data(dot_data)

#show graph
Image(graph.create_png())


# In[ ]:


#create pdf
graph.write_pdf("iris.pdf")


# ### Scatter Plot

# In[ ]:


# By selecting two features SepalLengthCm and SepelWidthCm
sns.FacetGrid(dataset, hue="Species", size=6)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()
plt.title('Sepal Length Vs Sepel Width')
plt.show()


# **Observations**
# 1. Using SepalLengthCm and SepalWidthCm features, we can distinguish Iris-setosa flowers from others.
# 2. Seperating Iris-versicolor from Iris-viginica is much harder as they have considerable overlap.

# In[ ]:


# By selecting two features PetalLengthCm and PetalWidthCm
sns.FacetGrid(dataset, hue="Species", size=6)    .map(plt.scatter, "PetalLengthCm", "PetalWidthCm")    .add_legend()
plt.title('Petal Length Vs Petal Width')
plt.show()


# **Observations**
# 1. Using PetalLengthCm and PetalWidthCm features, we can distinguish Iris-setosa flowers from others.
# 2. PetalLengthCm and PetalWidthCm features selection is giving better results than SepalLengthCm and SepalWidthCm features selection.

# ###  Pair Plot

# In[ ]:


sns.pairplot(dataset, hue="Species", size=3);
plt.show()


# **Observations**
# 1. PetalLengthCm and PetalWidthCm are the best features to identify various flower types.
# 2. Iris-setosa can be easily identified,Iris-virginica and Iris-versicolor have some overlap.

# ### Box Plot

# In[ ]:


plt.figure(figsize=(14,12))
plt.subplot(2,2,1)
sns.boxplot(x='Species', y = 'SepalLengthCm', data=dataset)
plt.subplot(2,2,2)
sns.boxplot(x='Species', y = 'SepalWidthCm', data=dataset)

plt.subplot(2,2,3)
sns.boxplot(x='Species', y = 'PetalLengthCm', data=dataset)
plt.subplot(2,2,4)
sns.boxplot(x='Species', y = 'PetalWidthCm', data=dataset)


# ### Violin Plots

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species', y = 'SepalLengthCm', data=dataset)
plt.subplot(2,2,2)
sns.violinplot(x='Species', y = 'SepalWidthCm', data=dataset)

plt.subplot(2,2,3)
sns.violinplot(x='Species', y = 'PetalLengthCm', data=dataset)
plt.subplot(2,2,4)
sns.violinplot(x='Species', y = 'PetalWidthCm', data=dataset)


# ## Clasification Algorithms

# In[ ]:


dataset.shape


# In[ ]:


dataset.head()


# In[ ]:


X_p = dataset.iloc[:,[2, 3]].values
y = dataset.iloc[:, 4].values
X_s = dataset.iloc[:,[0, 1]].values


# In[ ]:


X_p.shape


# In[ ]:


X_s.shape


# In[ ]:


y.shape


# In[ ]:


# Encoding categorical data (IT16001480)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y
# Iris-setosa == 0
# Iris-versicolor == 1
# Iris-virginica == 2


# In[ ]:


# Splitting the dataset into the Training set and Test set (Petal Lenth vs Petal Width)
from sklearn.model_selection import train_test_split
X_trainp, X_testp, y_trainp, y_testp = train_test_split(X_p, y, test_size = 0.3, random_state = 0)


# In[ ]:


# Splitting the dataset into the Training set and Test set (Sepal Length vs Sepal Width)
from sklearn.model_selection import train_test_split
X_trains, X_tests, y_trains, y_tests = train_test_split(X_s, y, test_size = 0.3, random_state = 0)


# ## Decision Tree Classifier

# In[ ]:


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)


# **Petal Length vs Petal Width**`

# In[ ]:


classifier.fit(X_trainp, y_trainp)


# In[ ]:


# Predicting the Test set results
y_predp = classifier.predict(X_testp)
y_predp


# In[ ]:


# Measuring Accuracy
from sklearn import metrics
print('The accuracy of Decision Tree Classifier is: ', metrics.accuracy_score(y_predp, y_testp))


# In[ ]:


# Making confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_testp, y_predp))


# In[ ]:


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_trainp, y_trainp
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen', 'lightblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Decision Tree (Training set)')
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.legend()
plt.show()


# In[ ]:


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_testp, y_testp
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen', 'lightblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Decision Tree (Test set)')
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.legend()
plt.show()


# ** Sepal Length vs Sepal Width**

# In[ ]:


# Fitting Decision Tree classifier to the Training set
classifier.fit(X_trains, y_trains)


# In[ ]:


# Predicting the Test set results
y_preds = classifier.predict(X_tests)
y_preds


# In[ ]:


# Measuring Accuracy
from sklearn import metrics
print('The accuracy of the Decision Tree Classifier using Sepals is:', metrics.accuracy_score(y_preds, y_tests))


# In[ ]:


# Making confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_tests, y_preds))


# In[ ]:


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_trains, y_trains
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen', 'lightblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Decision Tree(Training set)')
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.legend()
plt.show()


# In[ ]:


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_tests, y_tests
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen', 'lightblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Decision Tree(Test set)')
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.legend()
plt.show()


# **Observations**
# 
#    1. The accuracy of the Decision Tree Classifier using Petals is: 95 %
#    2. The accuracy of the Decision Tree Classifier using Sepals is: 64 %
# 

# # Observations

# Using Petals over Sepal for training the data gives a much better accuracy.
