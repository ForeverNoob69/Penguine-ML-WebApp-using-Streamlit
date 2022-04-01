import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import numpy as np

st.write(""" # Penguins Prediction App
         
         This app uses heavy ML algorithm to predict Penguins
         
         
         """)
st.sidebar.header("User Input Parameter")
st.sidebar.markdown("""
                    [Example CSV input File](https://raw.githubusercontent.com/dataprofessor/code/master/streamlit/part3/penguins_example.csv)""")

uploaded_file = st.sidebar.file_uploader("Upload your File", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_fun():
        island = st.sidebar.selectbox("Island",["Biscoe","Dream","Torgersen"])
        sex = st.sidebar.selectbox("Sex",["MALE","FEMALE","."])
        bill_length = st.sidebar.slider("Bill Length",32.1,59.6,40.0)
        bill_depth = st.sidebar.slider("Bill Depth",13.1,21.5,17.2)
        flipper_length = st.sidebar.slider("Flipper Length",172.0,232.0,202.0)
        body_mass = st.sidebar.slider("Body Mass",2700.0,6300.0,4200.0)
        data = {"island":island,
                "sex":sex,
                "culmen_length_mm":bill_length,
                "culmen_depth_mm":bill_depth,
                "flipper_length_mm":flipper_length,
                "body_mass_g":body_mass}
        feature = pd.DataFrame(data,index=[0])
        return feature
    input_df = user_input_fun()







penguins_raw = pd.read_csv("E:\py\club badge\penguins_size.csv.")
penguins_raw = penguins_raw.dropna(axis = 0, how ='any', thresh = None, subset = None, inplace=False)
penguin = penguins_raw.drop(columns=["species"])
df = pd.concat([input_df,penguin],axis=0)
encode = ["sex","island"]

for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]
print(df.columns)

df = df[:1]
print(df.shape)


st.subheader("User input Feature: ")
if uploaded_file is not None:
    st.write(df)
else:
    st.write("Import a CSV File or input your data into the parameter.")
    st.write(df)
    
def input_fn(data):
    return tf.data.Dataset.from_tensor_slices(dict(data))
test_fn = lambda:input_fn(df)

def create_model(optimizer='adam', init='uniform'):
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16, input_dim=df.shape[1], kernel_initializer=init, activation='relu'))
    model.add(tf.keras.layers.Dense(3, kernel_initializer=init, activation='softmax'))
    # Compile model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer="adam", metrics=['accuracy'])
    return model

expected = ['Adelie', 'Chinstrap','Gentoo']
check_point_path = "training_1/cp.ckpt"
model = create_model()
model.load_weights(check_point_path)

prediction = model.predict(df)
pred = expected[np.argmax(prediction)]
st.header("Predicion is: ")
st.write(pred)
st.header("Prediction Probability(%) is: ")
st.write(pd.DataFrame(prediction*100,columns=expected)) 