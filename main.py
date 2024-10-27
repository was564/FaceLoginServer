import os
from flask import Flask, flash, request, redirect, url_for, render_template
from facial_recognition import find_face, register_face
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', '.zip'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

response_post_file = {
    "status": 200,
    "file_name": "None",
    "name": "None",
    "birth": "None"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/face/check', methods=['POST'])
def check_face():
    image_file_list = request.files.getlist("file['images']")

    for file in image_file_list:
        if file.filename == '':
            flash('No Image')
            return "No Image"

    for file in image_file_list:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            result, result_value = find_face(path)
            if result:
                response_post_file["status"] = 200
                response_post_file["file_name"] = file.filename
                response_post_file["name"] = result_value
            else:
                response_post_file["status"] = 400
                response_post_file["file_name"] = ''
                response_post_file["name"] = result_value

            os.remove(path)

    return response_post_file


@app.route('/face/register', methods=['POST'])
def register_member():
    image_file = request.files['images']
    name = request.form['name'].replace('"', "")
    birth = request.form['birth'].replace('"', "")

    if image_file.filename == '':
        flash('No Image')
        return "No Image"
    elif name == '' or birth == '':
        flash('No Info')
        return 'No Info'

    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(path)
        result, result_value = register_face(UPLOAD_FOLDER, filename, name, birth)
        if result:
            response_post_file["status"] = 200
            response_post_file["file_name"] = image_file.filename
            response_post_file["name"] = result_value
            response_post_file["birth"] = birth
        else:
            response_post_file["status"] = 400
            response_post_file["file_name"] = ''
            response_post_file["name"] = result_value

        os.remove(path)
        return response_post_file

if __name__ == "__main__":
    app.run(host='0.0.0.0')