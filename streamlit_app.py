# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as pc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score,plot_confusion_matrix

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
def preiction (m,RI, Na, Mg, Al, Si, K, Ca, Ba,Fe):
  glass_type=m.predict([[RI, Na, Mg, Al, Si, K, Ca, Ba,Fe]])
  if glass_type==1:return "building windows float processed"
  elif glass_type == 2:return "building windows non float processed"
  elif glass_type == 3:return "vehicle windows float processed"
  elif glass_type == 4:return "vehicle windows non float processed"
  elif glass_type == 5:return "containers"
  elif glass_type == 6:return "tableware"
  else: return "headlamp"
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df
st.title('Glass type predictor')
sliders=[]

glass_df = load_data()
for magicmagicmagic in glass_df.columns[0:-1]:
    sliders.append(st.sidebar.slider(magicmagicmagic,float(glass_df[magicmagicmagic].min()),float(glass_df[magicmagicmagic].max())))
# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']
st.set_option('deprecation.showPyplotGlobalUse', False)


# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
a=st.sidebar.title('Explorarty Data Anylsis')
aa=st.sidebar.checkbox('show raw data')
aaa=st.sidebar.header('scatter')
aaaa=st.sidebar.multiselect('x',['RI','Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
aaaaaaa=st.sidebar.header('magicx')
aaaaaaaa=st.sidebar.multiselect('xxx',['RI','Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
magicmultiselc=st.sidebar.multiselect('magic',['hist','box','count','pychart','heat','bearplot'])
magicselect=st.sidebar.selectbox('classifier',('svm','rfc','lr'))
predictbutton=st.sidebar.button('predict')
for i in aaaa:
    st.subheader(i,' and glass type')
    f=pc.scatter(glass_df,x=i,y='GlassType')
    st.plotly_chart(f)
for ii in magicmultiselc:
    for iii in aaaaaaaa:
        if ii =='hist':
            st.subheader([ii,' plot for ',iii,' coloumn'])
            plt.hist(glass_df[iii])
            st.pyplot()
        elif ii =='box':
            st.subheader([ii,' plot for ',iii,' coloumn'])
            sns.boxplot(glass_df[iii])
            st.pyplot()
        elif ii =='count':
            st.subheader([ii,' plot for ',iii,' coloumn'])
            sns.countplot(glass_df,x='GlassType')
            st.pyplot()
        elif ii =='pychart':
            st.subheader([ii,' plot for ',iii,' coloumn'])
            plt.pie(glass_df['GlassType'].value_counts())
            st.pyplot()
        elif ii =='heat':
            st.subheader([ii, ' plot for ', iii, ' coloumn'])
            sns.heatmap(glass_df.corr(),annot=True)
            st.pyplot()
        elif ii =='bearplot':
            st.subheader([ii,' plot for ',iii,' coloumn'])
            sns.pairplot(glass_df)
            st.pyplot()
if aa:
    st.subheader('full data set')
    st.dataframe(glass_df)
if magicselect=='svm':
    c=st.sidebar.number_input('c value', 0.01, 100.00,step=0.01)
    g=st.sidebar.number_input('gamma value', 0.01, 100.00, step=0.01)
    k=st.sidebar.radio('kernel', ('linear','poly','rbf'))
    if predictbutton:
        model=SVC(kernel=k,C=c,gamma=g)
        model.fit(X_train,y_train)
        st.header(('model score',model.score(X_test,y_test)))
        plot_confusion_matrix(model,X_test,y_test)
        st.pyplot()
        st.header(('glass type is ',preiction(model,sliders[0],sliders[1],sliders[2],sliders[3],sliders[4],sliders[5],sliders[6],sliders[7],sliders[8])))
elif magicselect=='rfc':
    nest=st.sidebar.number_input('nest value', 100, 5000,step=10)
    maxdepths=st.sidebar.number_input('maxdepth value', 1, 100, step=1)
    if predictbutton:
        model=RandomForestClassifier(n_estimators=nest,max_depth=maxdepths,n_jobs=-1)
        model.fit(X_train,y_train)
        st.header(('model score',model.score(X_test,y_test)))
        plot_confusion_matrix(model,X_test,y_test)
        st.pyplot()
        st.header(('glass type is ',preiction(model,sliders[0],sliders[1],sliders[2],sliders[3],sliders[4],sliders[5],sliders[6],sliders[7],sliders[8])))
elif magicselect=='lr':
    c=st.sidebar.number_input('c value',1,100,step=1)
    maxit = st.sidebar.number_input('max iteration value', 10, 1000, step=10)
    if predictbutton:
        model=LogisticRegression(C=c,max_iter=maxit)
        model.fit(X_train,y_train)
        st.header(('model score', model.score(X_test, y_test)))
        plot_confusion_matrix(model, X_test, y_test)
        st.pyplot()
        st.header(('glass type is ',preiction(model, sliders[0], sliders[1], sliders[2], sliders[3], sliders[4], sliders[5], sliders[6],sliders[7], sliders[8])))
