import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('E-commerce Dataset EDA')

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('ecommerce_dataset.csv')
    return df

df = load_data()

st.header('Dataset Overview')
st.write(df.head())

st.header('Missing Values')
missing_values = df.isnull().sum()
st.write(missing_values)

st.header('Descriptive Statistics')
numerical_cols = df.select_dtypes(include=['int64', 'float64'])
descriptive_stats = numerical_cols.describe()
st.write(descriptive_stats)

st.header('Categorical Feature Analysis')
categorical_cols = ['category', 'region', 'payment_method']
for col in categorical_cols:
    st.subheader(f'Value counts for {col}:')
    st.write(df[col].value_counts())

st.header('Visualizations')

# Distribution of Categorical Features
st.subheader('Distribution of Categorical Features')
fig1, axes1 = plt.subplots(nrows=1, ncols=len(categorical_cols), figsize=(18, 6))
for i, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, ax=axes1[i])
    axes1[i].set_title(f'Distribution of {col}')
    axes1[i].set_xlabel(col)
    axes1[i].set_ylabel('Count')
    axes1[i].tick_params(axis='x', rotation=45)
st.pyplot(fig1)

# Distribution of Numerical Features
st.subheader('Distribution of Numerical Features')
numerical_cols = ['quantity', 'price', 'discount']
fig2, axes2 = plt.subplots(nrows=1, ncols=len(numerical_cols), figsize=(18, 6))
for i, col in enumerate(numerical_cols):
    sns.histplot(data=df, x=col, kde=True, ax=axes2[i])
    axes2[i].set_title(f'Distribution of {col}')
    axes2[i].set_xlabel(col)
    axes2[i].set_ylabel('Frequency')
st.pyplot(fig2)

# Relationship between Price and Discount
st.subheader('Relationship between Price and Discount')
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='price', y='discount', ax=ax3)
ax3.set_title('Relationship between Price and Discount')
ax3.set_xlabel('Price')
ax3.set_ylabel('Discount')
st.pyplot(fig3)

# Boxplot of Price
st.subheader('Boxplot of Price')
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df['price'], ax=ax4)
ax4.set_title('Boxplot of Price')
ax4.set_xlabel('Price')
st.pyplot(fig4)

st.header('Correlation Matrix of Numerical Features')
numerical_cols = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numerical_cols.corr()
fig5, ax5 = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax5)
ax5.set_title('Correlation Matrix of Numerical Features')
st.pyplot(fig5)