from flask import Flask, jsonify, request, Response
from flask_pymongo import PyMongo
from bson import json_util
from bson.objectid import ObjectId
from ocr import get_chars

app = Flask(__name__)
app.secret_key = 'myawesomesecretkey'
app.config['MONGO_URI'] = 'mongodb+srv://chat_user:chat_user123@uptask-mern.sb1cxkm.mongodb.net/virtual-classroom-images'
mongo = PyMongo(app)
dbName = ['virtual-classrooms-images']


@app.route('/get_chars_image/<_id>', methods=['PUT'])
def get_chars_from_image(_id):
    img = mongo.db.images.find({'_id': ObjectId(_id)})

    try:
        text = get_chars('1')
        print(text)
    except:
        response = jsonify({'msg': 'Ocurrio un error con la Imagen: ' + id })
        response.status_code = 500
        return response

    mongo.db.images.update_one(
        {'_id': _id},
        {'$set': {'text': 'editado'}})

    response = jsonify({'msg': 'Carácteres de la Imagen: ' + _id + ' se han obtenido exitosamente'})
    response.status_code = 200
    return response


if __name__ == "__main__":
    app.run(debug=True, port=3000)
