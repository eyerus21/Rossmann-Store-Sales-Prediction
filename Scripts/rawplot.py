import streamlit as st
#import awesome_streamlit as ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Plots ..."):
        st.title('Raw Data Visualisation  📈 📊')

        # read the datasets
        na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        train = pd.read_csv('../data/train.csv', na_values=na_value)
        store = pd.read_csv('../data/store.csv', na_values=na_value)
        full_train = pd.merge(left = train, right = store, how = 'inner', left_on = 'Store', right_on = 'Store')
        st.sidebar.title("Gallery")
        st.sidebar.subheader("Choose Feature or Aspect to plot")
        plot = st.sidebar.selectbox("feature", ("Seasonality", "Correlation", "SchoolHoliday", "Open/DayOfWeek", 'Promotions', 'State Holiday', 'PromoIntervals', 'Assortment', 'Store Type','Competition'))

        # SchoolHoliday plots
        if plot == 'SchoolHoliday':
            st.subheader("School Holidays")
            sns.countplot(x='SchoolHoliday', data=full_train, palette = 'Set2').set_title('a count plot of school holidays')
            st.pyplot()
            fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

            sns.barplot(x='SchoolHoliday', y='Sales', data=full_train, ax=axis1, palette = 'Set2').set_title('sales across ordinary school days and school holidays')
            sns.barplot(x='SchoolHoliday', y='Customers', data=full_train, ax=axis2, palette = 'Set2').set_title('no of customers across ordinary school days and school holidays')
            st.pyplot()
            # st.write("""
            # Not so many stores were affected by the closure of schools.
            # But for the few affected, their sales don't 
            # """)
