import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
sns.set_theme(style="darkgrid")


st.set_page_config(
   page_title="🤖 Camilo Franco",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

with st.container():
    col1, col2 = col1, col2 = st.columns([3,2],gap='small')
    with col1:
        st.write('')
        st.write('')
        st.write('')
        st.header("Breast cancer prediction via clustering⚕️")
        st.markdown("Camilo Franco : Data Scientist")
        st.markdown('''- Github [@camigenius](https://github.com/camigenius)''')
    with col2:
        image = Image.open('cluster2.jpg')
        st.image(image, caption='Foto Unplash,Autor: Pierre Bamin, https://unsplash.com/license')    

df = pd.read_csv('dataset_wisc_sd.csv')
df = df.replace(r'\\n','', regex=True)
df = df.dropna()
df = df
size = df.shape
Features = df.columns


st.markdown(f"Size Data : {size[0]}")
view_features = st.expander('Features')

with view_features:
    st.write(df.columns)


st.table(df.head())

st.subheader("Diagnosis :  M for malignant and B for benign")


fig = px.histogram(df, x="diagnosis",color='diagnosis', height=400)
st.plotly_chart(fig)

st.subheader("Pairplot")
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
select_feats =['diagnosis','radius_mean','texture_mean','smoothness_mean']
ax_sb = sns.pairplot(df[select_feats],hue='diagnosis',markers=['s','o'])
st.pyplot(ax_sb)

scaler = StandardScaler()
X = df.drop(columns=['id','diagnosis'])
y = df['diagnosis'].values
X_scaled = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)

view_scaled = st.expander('Features Scaled-StandarScaler')
with view_scaled:
    st.table(X_scaled.head())

agc = AgglomerativeClustering(n_clusters=2,linkage='ward')
agc_featAll_pred= agc.fit_predict(X_scaled.iloc[:,:2])


#plt.figure(figsize=(20,5))
#plt.subplot(121)
st.write("---")
st.header("Agglometative Clustering Model")
st.markdown("Scatter plot before and after training")
st.text("Notice : parametrer(linkage = ‘ward’) minimizes the variance of the clusters being merged.")


code ='''
agc = AgglomerativeClustering(n_clusters=2,linkage='ward')
agc_featAll_pred= agc.fit_predict(X_scaled.iloc[:,:2])

'''
st.code(code, language = 'python')

with st.container():
        col1,col2=st.columns([2,2],gap='small')
        
        with col1:
            fig_sb1 , ax_sb1 = plt.subplots(figsize=(20,7))
            plt.subplot(121)
            plt.title('Actual Results')
            ax_sb1 = sns.scatterplot(x='radius_mean',y='texture_mean',
                                hue = y,
                                style = y,
                                data=X_scaled,
                                markers=['s','o'],                                
                                )
            st.pyplot(fig_sb1)

        with col2:
            fig_sb2 , ax_sb3 = plt.subplots(figsize=(20,7))
            #ax.legend(loc='upper right')
            plt.subplot(122)
            plt.title("Predict Agglomerative Clustering")
            ax_sb3 = sns.scatterplot(x='radius_mean',y='texture_mean',
                                hue=agc_featAll_pred,
                                style=agc_featAll_pred,
                                data = X_scaled,
                                markers = ['s','o'])
            #ax.legend(loc = 'upper right')

            st.pyplot(fig_sb2)

accuracy = accuracy_score(agc_featAll_pred,y)
st.subheader('Accuracy : %.2f ' %  accuracy_score(agc_featAll_pred,y))

# 'Kmeans- First Transform PCA '
st.write("---")
st.header("K-means Model")
st.subheader('Dimensionality Reduction(PCA)-Principal Components Analysis " ')



pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
kmc = KMeans(n_clusters=2, n_init=10, init="k-means++")
kmc_predict = kmc.fit_predict(X_pca)

code ='''
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
kmc = KMeans(n_clusters=2, n_init=10, init="k-means++")
kmc_predict = kmc.fit_predict(X_pca)
'''
st.code(code, language = 'python')



print(accuracy_score(kmc_predict,y))


df['Predict']=  kmc_predict = kmc.fit_predict(X_pca)
st.dataframe(df[['diagnosis','Predict']].sample(n = 25).T)

df_pca = pd.DataFrame(X_pca,columns=['PCA1','PCA2'])
df_pca['y'] = y

st.dataframe(df_pca[['PCA1','PCA2']].head(n = 18).T)
ax_sb4 = sns.relplot(data= df_pca, x = 'PCA1', y = 'PCA2', hue = 'y' ,height=8, aspect=15/8 )
st.pyplot(ax_sb4)


st.subheader('Accuracy : %.2f ' %  accuracy_score(kmc_predict,y))


st.write("🐺Camilo Franco🐺")
st.markdown('''- Github [@camigenius](https://github.com/camigenius)''')

st.markdown('''- Linkedin [@camilofrancog](https://www.linkedin.com/in/camilofrancog/)''')