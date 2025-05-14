import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import pickle as pk
import json
from streamlit_lottie import st_lottie
import numpy as np
import plotly.express as px
pg_bg_img="""
<style>
[data-testid="stAppViewContainer"] {
background-image: url("crop.png");
background-size: cover;
}
</style>
"""

st.set_page_config(page_title='CROP_RECOMMENDATION_SYSTEM',layout='wide',page_icon='🌱')


attribution="""

- N - ratio of Nitrogen content in soil
- P - ratio of Phosphorous content in soil
- K - ratio of Potassium content in soil
- temperature - temperature in degree Celsius
- humidity - relative humidity in %
- ph - ph value of the soil
- rainfall - rainfall in mm
"""

st.markdown(pg_bg_img,unsafe_allow_html=True)

#horizontal menu :
selected=option_menu(
        menu_title='🌾CROP RECOMMENDATION SYSTEM¶🌾',
        options=['inf','Dashboard','predict'],
        icons=['info-circle-fill','person-circle'],
        menu_icon='🌾',
        orientation='horizontal',
        default_index=1,
        styles={
        "container": {"padding": "5!important","background-color":'#6AC6F7'},
        "icon": {"color": "white", "font-size": "23px"}, 
        "nav-link": {"color":"black","font-size": "20px", "text-align": "center", "margin":"0px", "--hover-color": "#8EF314"},
        "nav-link-selected": {"background-color": "white"},
            }                 
    )

if selected=='predict':
    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color:#6AC6F7;
        color:#ffffff;
    }
    div.stButton > button:hover {
        background-color: #0C0C0C ;
        color:#ff99ff;
        }
    </style>""", unsafe_allow_html=True)
    model=pk.load(open('model.pkl','rb'))
    with st.expander('Data fields'):
        st.markdown(attribution)
    p1,p2=st.columns(2)
    with p1:
        
        pp1,pp2,pp3=st.columns(3)
            
        with pp1:
            N=st.number_input('Nitrogen(N)',0,140)
            P=st.selectbox('Phosphorous(P)',range(5,145))
            K=st.selectbox('Potassium(K)',range(5,205))  
             
        with pp2:
            temperature=st.selectbox('Temperature',range(8,44)) 
            humidity=st.number_input('humidity',14,100)
            ph=st.selectbox('ph',range(3,10))
            
        with pp3:
            rainfall=st.selectbox('rainfall',range(20,299))
            if st.button('predict.'):
                with p2:
                    
                    
                    input_data_module=pd.DataFrame([[N,P,K,temperature,humidity,ph,rainfall]],
                    columns=['N','P','K','temperature','humidity','ph','rainfall']
                    )
                
                    with st.expander('View you selected Soil nutrien values'):
                        st.dataframe(input_data_module)
                    l1,l2=st.columns(2)
                    crop=model.predict(input_data_module)
                    if crop==1:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Rice 
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('1.png',width=180)    
                    elif crop==2:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Maize 
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('2.PNG',width=180)    
                    elif crop==3:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Chickpea 
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('3.png',width=180)    
                    elif crop==4:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Kidneybeans 
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('4.png',width=180)    
                    elif crop==5:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Pigeonpeas 
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('5.png',width=180)    
                    elif crop==6:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Mothbeans 
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('6.png',width=180)    
                    elif crop==7:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Mungbean
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('7.png',width=180)    
                    elif crop==8:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Blackgram
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('8.png',width=180)    
                    elif crop==9:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Lentil
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('9.png',width=180)    
                    elif crop==10:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Pomegranate
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('10.png',width=180)    
                    elif crop==11:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Banana
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('11.png',width=180)    
                    elif crop==12:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Mango
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('12.png',width=180)    
                    elif crop==13:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Grapes
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('13.png',width=180)    
                    elif crop==14:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Blackgram
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('14.png',width=180)    
                    elif crop==15:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Muskmelon
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('15.png',width=180)    
                    elif crop==16:
                        st.write('apple')
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Apple
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('16.png',width=180)    
                    elif crop==17:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Orange
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('17.png',width=180)    
                    elif crop==18:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Papaya
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('18.png',width=180)    
                    elif crop==19:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Coconut
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('19.png',width=180)    
                    elif crop==20:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Cotton
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('20.png',width=180)    
                    elif crop==21:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - jute
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('21.png',width=180)    
                    elif crop==22:
                        l1.markdown("""
                                    Recommend Crop...         
                                    - Coffee
                                    Is the best crop to Be 
                                    cultivated right there""")
                        l2.image('22.png',width=180)    



if selected=='inf':
    with st.container():
        st.header('🎯 Objective')
        st.markdown("""
                    - Our project aims to create a smale Crop Recommendation System that uses soil nutrient data to suggest the best crops for a specifc farming area . 
                    - By using advanced machine learning . we want to help farmers choose the right crops. boosting productivity and sustainability .
                    - Our goal is to empower farmers with accurate recommendations for better harvests and efficient resource use. 
                    """)
    
    i1,i2=st.columns(2)
    with i1:
        st.subheader('🚀 Project Highlights:')
        st.markdown("""
                
                    - 🌿 Dataset: We've gathered comprehensive soil nutrient data, setting the foundation for our analysis.
                    - 🧐 Exploratory Data Analysis (EDA): Insights gained from EDA guided our understanding of the data's patterns and outliers.
                    - 🧹 Data Preprocessing: We meticulously cleaned and organized the data to ensure accuracy and reliability.
                    - ⚖️ Normalization: All data was standardized to enable accurate comparisons and analysis.
                    - 💡 Modeling: Advanced machine learning techniques were applied to build our intelligent recommendation system.
                    - 🌟 Results: Our efforts culminated in accurate crop recommendations that can empower farmers to enhance productivity and sustainability.

                    """)
        st.subheader('Conclusion...')
        st.markdown("""
                    - This machine learning model has been developed and traing historical data.
                    - As we can see RandomForestClassifier performing best (with accuracy 0.97)
                    - Model	                        Score
                    - 1	RandomForestClassifier	     99.31
                    - 2	Support Vector Classifer	 97.66
                    - 3	Desision_Tree_classification 97.66
                    - 4	KNeighborsClassifier	     97.38
                    - 5	Logistic_Regression	         92.29

                    """)
    with i2:
        lin_butt="""
        <style>
        .st-emotion-cache-1mcbg9u {
        background-color: #6AC6F7;
        }
        </style>
        
        """
        st.markdown(lin_butt,unsafe_allow_html=True)
        st.link_button('About-Me🌿','https://www.linkedin.com/in/makhan-kumbhkar-44b5361a/')
        def get(path:str):
            with open(path,'r') as p:
                return json.load(p)
        path=get('./nani.json')
        st_lottie(path,height=300,width=600)


if selected=='Dashboard':

    cpro=pd.read_csv('crop_production.csv')

    c1,c2,c3,c4,c5,c6,c7=st.columns(7)

    with c5:
        state_name=st.selectbox('State Name',cpro['State_Name'].unique())
    with c6:
        crop1=st.selectbox('Crop',cpro['Crop'].unique())
    with c7:
        district_name=st.selectbox('District Name',cpro['District_Name'].unique())

    with c1:
        with st.container(border=True):
            st.write('Number of Crops')
            rce=cpro['Crop'].nunique()
            st.subheader(rce)
    with c2:
        with st.container(border=True):
            st.write('Total Season')
            st.subheader(cpro['Season'].nunique())
    with c3:
        with st.container(border=True):
            st.write('Total State Name ')
            st.subheader(cpro['State_Name'].nunique())
    with c4:
        with st.container(border=True):
            st.write('Total District Name')
            st.subheader(cpro['District_Name'].nunique())


    s1,s2,s3=st.columns(3)
    with s1:
        with st.container(border=True):
            std_wise_crop=cpro.groupby(by='State_Name')['Crop'].value_counts().reset_index()
            std_input=std_wise_crop[std_wise_crop['State_Name']==state_name]
            fig = px.pie(std_input, values='count', names='Crop', title='State '+state_name+' Wise Crop',color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig)

    with s3:
        with st.container(border=True):
            dis_wise_crop=cpro.groupby(by='District_Name')['Crop'].value_counts().reset_index()
            dis_input=dis_wise_crop[dis_wise_crop['District_Name']==district_name]
            fig = px.pie(dis_input, values='count', names='Crop', title='District '+district_name+' Wise Crop')
            st.plotly_chart(fig)

    with s2:
        with st.container(border=True):
            cro_wise_prod=cpro.groupby(by='Crop')['State_Name'].value_counts().reset_index()
            crop_input=cro_wise_prod[cro_wise_prod['Crop']==crop1]
            fig = px.bar(crop_input, x='State_Name', y='count',title='Crop ' +crop1+' Wise State name')
            st.plotly_chart(fig)
    v1,v2=st.columns([0.35,0.65])
    with v1:
        with st.container(border=True):
            
            Season_wise=cpro.groupby(by='Season')['Production'].sum().reset_index()
            # st.bar_chart(Season_wise, y="Production", x="Season", horizontal=True)
            fig=px.bar(Season_wise, y='Season', x='Production',title='Season Wise Production')
            st.plotly_chart(fig)
    with v2:
        with st.container(border=True):
            
            year_wise=cpro.groupby(by='Crop_Year')['Production'].sum().reset_index()
            # st.bar_chart(Season_wise, y="Production", x="Season", horizontal=True)
            fig=px.line(year_wise, x='Crop_Year', y='Production',title='Year Wise Production',markers=True)
            st.plotly_chart(fig)

    h1,h2,h3,h4=st.columns(4)
    crop=pd.read_csv('Crop.csv')
    with h1:
        with st.container(border=True):
            k_wise_crop=crop[['K','label']].groupby(by='label').value_counts().reset_index()
            fig = px.bar(k_wise_crop, y="label", x="K",title='Potassium content in soil wise crop')
            st.plotly_chart(fig)

    with h2:
        with st.container(border=True):
            p_wise_crop=crop[['P','label']].groupby(by='label').value_counts().reset_index()
            fig = px.bar(p_wise_crop, x="label", y="P",title='Phosphorous content in soil wise crop',
                         labels={'P':'Phosphorous','label':'Crop'})
            st.plotly_chart(fig)
    with h3:
        with st.container(border=True):
            tem_wise_crop=crop[['temperature','label']].groupby(by='label').value_counts().reset_index()
            fig = px.bar(tem_wise_crop, x="label", y="temperature",title='temperature in degree Celsius wise crop')
            st.plotly_chart(fig)
    with h4:
        with st.container(border=True):
            n_wise_crop=crop[['N','label']].groupby(by='label').value_counts().reset_index()
            fig = px.bar(n_wise_crop, y="label", x="N",title='Nitrogen content in soil wise crop')
            st.plotly_chart(fig)
    p1,p2=st.columns([0.35,0.65])
    with p1:
        with st.container(border=True):
            h_wise_crop=crop[['humidity','label']].groupby(by='label').value_counts().reset_index()
            fig = px.bar(h_wise_crop, y="label", x="humidity",title='Relative humidity wise crop')
            st.plotly_chart(fig)
    with p2:
        with st.container(border=True):
            ph_wise_crop=crop[['ph','label']].groupby(by='label').value_counts().reset_index()
            fig = px.bar(ph_wise_crop, y="ph", x="label",title='Relative humidity wise crop')
            st.plotly_chart(fig)
    
    with st.container(border=True):
        rain_wise_crop=crop[['rainfall','label']].groupby(by='label').value_counts().reset_index()
        fig = px.line(rain_wise_crop, y="rainfall", x="label",title='rainfall wise crop')
        st.plotly_chart(fig)
st.markdown('_____________________________________________________')

            
