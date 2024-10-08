from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

model_filename = 'cancer_prediction_model.joblib'
loaded_model = joblib.load(model_filename)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_file', methods=['POST'])
def process_file():
    
    file_path = request.form['file-path']

    try:
        
        df = pd.read_csv(file_path)

        
        target_column = 'Diagnosis'

        
        X = df.drop(columns=[target_column])
        y = df[target_column]

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred_loaded = loaded_model.predict(X_test) 

       
        mse = mean_squared_error(y_test, y_pred_loaded)
        loaded_model_accuracy = accuracy_score(y_test, y_pred_loaded)

       
        result_data = {
            'mse': mse,
            'accuracy': loaded_model_accuracy * 100, 
            'predictions': y_pred_loaded.tolist()  
        }

       
        return render_template('result.html', result=result_data)

    except FileNotFoundError:
       
        return render_template('error.html', message='File not found.')

    except Exception as e:
        
        return render_template('error.html', message=f'An error occurred: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)