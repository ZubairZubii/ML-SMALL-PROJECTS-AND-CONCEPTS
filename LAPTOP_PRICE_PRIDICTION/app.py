from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

@app.route('/')
def index():
    # Pass data to the front-end (HTML) to populate select options
    brands = df['Company'].unique()
    types = df['TypeName'].unique()
    cpus = df['Cpu_Brand'].unique()
    gpus = df['Gpu brand'].unique()
    os_types = df['os'].unique()

    return render_template('index.html', brands=brands, types=types, cpus=cpus, gpus=gpus, os_types=os_types)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    company = request.form['brand']
    type = request.form['type']
    ram = int(request.form['ram'])
    weight = float(request.form['weight'])
    
    # Convert touchscreen and ips to 1 or 0
    touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
    ips = 1 if request.form['ips'] == 'Yes' else 0
    
    screen_size = float(request.form['screen_size'])
    resolution = request.form['resolution']
    cpu = request.form['cpu']
    hdd = int(request.form['hdd'])
    ssd = int(request.form['ssd'])
    gpu = request.form['gpu']
    os = request.form['os']

    # Calculate PPI based on screen resolution
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2)) ** 0.5 / screen_size

    # Prepare the query for model prediction
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, 12)
    
    # Predict price
    predicted_price = int(np.exp(pipe.predict(query)[0]))

    # Pass all form data along with the predicted price to the result page
    return render_template('result.html', 
                           brand=company, 
                           type=type, 
                           ram=ram, 
                           weight=weight, 
                           touchscreen='Yes' if touchscreen == 1 else 'No', 
                           ips='Yes' if ips == 1 else 'No', 
                           screen_size=screen_size, 
                           resolution=resolution, 
                           cpu=cpu, 
                           hdd=hdd, 
                           ssd=ssd, 
                           gpu=gpu, 
                           os=os, 
                           price=predicted_price)


if __name__ == '__main__':
    app.run(debug=True)
