from flask import Flask, jsonify, request, Response
from flask_pymongo import PyMongo
from bson import json_util
from bson.objectid import ObjectId
from ocr import get_chars

app = Flask(__name__)
app.secret_key = 'myawesomesecretkey'
app.config['MONGO_URI'] = 'mongodb+srv://chat_user:chat_user123@uptask-mern.sb1cxkm.mongodb.net/virtual-classrooms-images'
# cluster = MongoClient('mongodb+srv://chat_user:chat_user123@uptask-mern.sb1cxkm.mongodb.net/virtual-classroom-images')
# db = cluster['virtual-classroom-images']
# collection = db['images']
mongo = PyMongo(app)


@app.route('/get_chars_image/<_id>', methods=['PUT'])
def get_chars_from_image(_id):
    img = mongo.db.images.find_one({'_id': ObjectId(_id), })

    print(img["url"])
    #try:
    text = get_chars(img["url"])
    print(text)
    #except:
    #   response = jsonify({'msg': 'Ocurrio un error con la Imagen: ' + _id})
    #   response.status_code = 500
    #   return response

    mongo.db.images.update_one(
        {'_id': ObjectId(_id['$oid']) if '$oid' in _id else ObjectId(_id)},
        {'$set': {'text': 'text'}})

    response = jsonify({'msg': 'Carácteres de la Imagen: ' + _id + ' se han obtenido exitosamente'})
    response.status_code = 200
    return response


if __name__ == "__main__":
    app.run(debug=True, port=3000)
