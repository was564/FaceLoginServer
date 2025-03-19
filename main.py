import os
from flask import Flask, flash, request
from facial_recognition import find_face, register_face, check_member

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', '.zip'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

response_post_file = {
    "status": 200,
    "name": "None",
    "birth": "None",
    "try": -1
}

def init_post_file():
    response_post_file["status"] = 200
    response_post_file["name"] = "None"
    response_post_file["birth"] = "None"
    response_post_file["try"] = -1

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/face/check', methods=['POST'])
def check_face():
    init_post_file()
    image_file_list = request.files.getlist("images")

    for file in image_file_list:
        if file.filename == '':
            flash('No Image')
            return "No Image"

    saved_file_list = []
    result_list = []
    for file in image_file_list:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            saved_file_list.append(path)
            (result, result_value) = find_face(path)
            result_list.append((result, result_value))

    name_count = dict()
    max_tuple = (0, "nothing")
    error_count = (0, "error_name")
    for hasWho, name in result_list:
        if hasWho is False:
            error_count = (error_count[0] + 1, name)
        if name_count.get(name) is None:
            name_count[name] = 1
        else:
            name_count[name] += 1

        if name_count[name] > max_tuple[0]:
            max_tuple = (name_count[name], name)

    if max_tuple[0] < int(len(result_list) * 0.5 + 1) | error_count[0] > int(len(result_list) * 0.5):
        response_post_file["status"] = 400
        response_post_file["try"] = max_tuple[0]
        response_post_file["name"] = max_tuple[1]
    else:
        response_post_file["status"] = 200
        response_post_file["try"] = max_tuple[0]
        response_post_file["name"] = max_tuple[1]
    for path in saved_file_list:
        os.remove(path)

    print(response_post_file)
    return response_post_file

@app.route('/face/delete_member', methods=['POST'])
def delete_member():
    init_post_file()
    name = request.form['name'].replace('"', "")
    birth = request.form['birth'].replace('"', "")


@app.route('/face/check_member', methods=['POST'])
def check_member_rq():
    init_post_file()
    name = request.form['name'].replace('"', "")
    birth = request.form['birth'].replace('"', "")

    result, result_value = check_member(name, birth)
    if result:
        response_post_file["status"] = 200
    else:
        response_post_file["status"] = 400
        response_post_file["name"] = result_value

    return response_post_file


@app.route('/face/register', methods=['POST'])
def register_member():
    init_post_file()
    image_file_list = request.files.getlist('images')
    name = request.form['name'].replace('"', "")
    birth = request.form['birth'].replace('"', "")

    if len(image_file_list) == 0:
        flash('No Image')
        return "No Image"
    elif name == '' or birth == '':
        flash('No Info')
        return 'No Info'

    saved_file_list = []
    for i in range(0, len(image_file_list)):
        file = image_file_list[i]
        file.filename = name + '_' + birth + '_' + str(i) + ".jpg"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            saved_file_list.append(path)

    result, result_value = register_face(saved_file_list, name, birth)
    if result:
        response_post_file["status"] = 200
        response_post_file['birth'] = birth
        response_post_file["name"] = name
    else:
        response_post_file["status"] = 400

    for path in saved_file_list:
        os.remove(path)

    return response_post_file


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
