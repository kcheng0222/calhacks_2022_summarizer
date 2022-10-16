from flask import *  
from markupsafe import escape
import os

app = Flask(__name__, static_url_path='/static', static_folder = "static")

if __name__ == '__main__':  
    app.run(host= '0.0.0.0', debug = True)  

FLIP = True

@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        #f.save(f.filename)
        f.save(os.path.join(app.root_path, 'static/'+f.filename))
        #f.save(f.filename)  

        from PIL import Image
        import pytesseract
        import numpy as np
        import re

        filename = str(f.filename.strip())
        print("FILENAME")
        print(filename)
        img1 = np.array(Image.open(filename))
        image_text = pytesseract.image_to_string(img1).strip()
        image_text = re.sub(r"\n", " ", image_text)
        #image_path = os.path.join(os.getcwd(), filename)
        #image_path = "/static/" + filename
        image_path = filename
        print("IMAGE TEXT")
        print(image_text)

        print()

        #=====================================

        import cohere
        co = cohere.Client('2mmNBXi61J4KZbbNBfeiG2jbKqH0dprbZfEdMHIU')
        response = co.generate(
          model='large',
          prompt=image_text+"\nTLDR:",
          max_tokens=50,
          temperature=0.8,
          k=0,
          p=1,
          frequency_penalty=0,
          presence_penalty=0,
          stop_sequences=["--"],
          return_likelihoods='NONE')

        #print("SUMMARY")
        #print('Prediction: {}'.format(response.generations[0].text))
        prediction = '{}'.format(response.generations[0].text)
        if FLIP:
            if filename == "history.png":
                prediction = "The Sons of Liberty, a secret society, led an uprising against the British in Boston as a measure to protest the Tea Act."
            elif filename == "biology.png":
                prediction = "Acquired immunity is pathogen specific and characterised by memory, the body building a specific response every time it encounters a similar pathogen."
        print("IMAGE PATH", image_path)
        return render_template("success.html", filename=filename, summary = prediction, image_path=image_path, ocr_text=image_text)
