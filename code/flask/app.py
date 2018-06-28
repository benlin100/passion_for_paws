from flask import Flask, request, render_template
import pickle
import numpy as np
import json
import xgboost as xgb
import pandas as pd
import json
from xgboost.sklearn import XGBClassifier
from werkzeug.datastructures import ImmutableMultiDict

fields = ['animal_type_Dog', 'intake_type_Public Assist', 'intake_type_Stray', 'Intake_Neutered_Neutered',               'Intake_Sex_Male', 'col_White', 'col_Black', 'col_Grey', 'col_Yellow', 'col_Red', 'col_Blue', 'col_Tricolor', 'col_Brown', 'col_Orange', 'breed_Sporting', 'breed_Hound', 'breed_Working', 'breed_Terrier', 'breed_Toy', 'breed_Non_Sporting', 'breed_Herding', 'breed_Longhaired', 'breed_Mediumhaired', 'breed_Shorthaired', 'sqrt_age']



xgb = XGBClassifier()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/getdelay',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result=request.form
        

        num_values = len(result.getlist(fields[0]))
        data = []
        for i in range(num_values):
            data.append({field: result.getlist(field)[i] for field in fields})
            
        data= pd.DataFrame(data)
        data = data.apply(pd.to_numeric, errors='ignore')
        data = data.replace('on', 1) #this is for checkboxes..i think lol 
        data = data.replace('off', 0) #this is for checkboxes..i think lol
        data = data[fields]
        print(data)
        
        
        #result = ImmutableMultiDict(result)
        #result = result.getlist('animal_type_Dog')
        
        #print(result)
        
        #have to turn results 
        #ImmutableMultiDict([('animal_type_Dog', '1'), ('intake_type_Public Assist', '1'), ('intake_type_Stray', '1'), ('Intake_Neutered_Neutered', '1'), ('Intake_Sex_Male', '1'), ('col_White', '1'), ('col_Black', '1'), ('col_Grey', '1'), ('col_Yellow', '1'), ('col_Red', '1'), ('col_Blue', '1'), ('col_Tricolor', '1'), ('col_Brown', '1'), ('col_Orange', '1'), ('breed_Sporting', '1'), ('breed_Hound', '1'), ('breed_Working', '1'), ('breed_Terrier', '0'), ('breed_Toy', '1'), ('breed_Non_Sporting', '1'), ('breed_Herding', '1'), ('breed_Longhaired', '1'), ('breed_Mediumhaired', '1'), ('breed_Shorthaired', '1'), ('sqrt_age', '1')])
        #into a pandas dataframe row
        
 

        #Prepare the feature vector for prediction
        #index_dict = pickle.load(open("cat","rb"))
        #new_vector = np.array(index_dict).reshape((1,-1))               
   
        
        model = pickle.load(open("xgbmodel.pkl","rb"))
        prediction = model.predict(data)
               
        return render_template('result.html',prediction=prediction)

    
if __name__ == 'main':
    app.run(debug=True)
    
    
    
    
    
    
    
    
    






