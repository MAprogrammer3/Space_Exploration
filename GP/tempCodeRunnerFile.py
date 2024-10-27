from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os  # For file operations
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import pandas as pd
8
# Configure Flask app
app = Flask(__name__)

# Define allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extension
def allowed_file(filename):
  return '.' in filename and \
         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getData(label):
  df = pd.read_excel('static/SpaceExploration_data.xlsx')
  data = df[label]
  return data


@app.route('/', methods=['GET', 'POST'])
def upload_image():
  if request.method == 'GET':
    # Render the upload.html template for initial page load
    return render_template('upload.html')
  else:
    # Handle file upload on POST request
    file = request.files['file']
    image_url = request.form.get('imageLink')
    
    if image_url:
        # Handle image URL download
        try:
          response = requests.get(image_url, stream=True)
          response.raise_for_status()  # Raise an exception for bad status codes
          image = 'ss'
          # Extract filename from URL
          filename = image_url.split('/')[-1]
          if response.status_code == 200:
                # Open the image and save it to the static folder
                image = Image.open(BytesIO(response.content))
                image_path = os.path.join('static', 'downloaded_image.png')
                image.save(image_path)

          # Continue with the prediction process using 'static/filename'
          labels = ['asteroids',
                    'earth',
                    'elliptical',
                    'jupiter',
                    'mars',
                    'mercury',
                    'moon',
                    'neptune',
                    'pluto',
                    'saturn',
                    'spiral',
                    'uranus',
                    'venus']
          
          image = tf.keras.utils.load_img(os.path.join('static', 'downloaded_image.png'), target_size=(180, 180))
          img_arr = tf.keras.utils.array_to_img(image)
          img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension
          #model = tf.keras.models.load_model(os.path.join('static','space.h5'))
          model = tf.keras.models.load_model('models/space.h5')

          prediction = model.predict(img_bat)
          score = tf.nn.softmax(prediction)
          prediction_text = labels[np.argmax(score)]
          planet_data = getData(prediction_text)
          if prediction_text:
            return render_template('result.html', filename=filename, image_url='downloaded_image.png', prediction=prediction_text, planet_data=planet_data)
          else:
            return render_template('upload.html', filename=filename, image_url=f'/{filename}', prediction='')
        except requests.exceptions.RequestException as e:
          return render_template('upload.html', message=f'Error downloading image: {e}')
    elif file and allowed_file(file.filename):
      
        # Handle file upload
        filename = secure_filename(file.filename)
        file.save(os.path.join('static', filename))
        labels = ['asteroids',
                'earth',
                'elliptical',
                'jupiter',
                'mars',
                'mercury',
                'moon',
                'neptune',
                'pluto',
                'saturn',
                'spiral',
                'uranus',
                'venus']
        image = tf.keras.utils.load_img(os.path.join('static', filename), target_size=(180, 180))
        img_arr = tf.keras.utils.array_to_img(image)
        img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension
        # model = tf.keras.models.load_model('models/modelSaved.keras')
        model = tf.keras.models.load_model('models/space.h5')

        prediction = model.predict(img_bat)
        score = tf.nn.softmax(prediction)
        prediction_text = labels[np.argmax(score)]
        planet_data = getData(prediction_text)
        if prediction_text:
          return render_template('result.html', filename=filename, image_url=f'/{filename}', prediction=prediction_text, planet_data=planet_data)
        else:
          return render_template('upload.html', filename=filename, image_url=f'/{filename}', prediction='')
    else:
        return render_template('upload.html', message='No file selected or image URL provided') 

@app.route('/result/<filename>')
def show_result(filename):
  # No logic needed here, redirect to upload page
  return redirect(url_for('upload_image'))  # Redirects to upload page

@app.route('/static/<path:filename>')
def serve_static(filename):
  return send_from_directory('static', filename)

if __name__ == '__main__':
  # Create uploads directory if it doesn't exist
  #  os.makedirs('uploads', exist_ok=True)  # Create uploads directory if missing
  app.run(debug=True)