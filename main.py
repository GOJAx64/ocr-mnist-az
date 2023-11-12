from flask import Flask, jsonify
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from flask_cors import CORS
from interface import get_chars


def create_app():
    app = Flask(__name__)
    app.secret_key = 'myawesomesecretkey'
    app.config[
        'MONGO_URI'] = 'mongodb+srv://chat_user:chat_user123@uptask-mern.sb1cxkm.mongodb.net/virtual-classrooms-images'
    CORS(app, origins=["https://virtual-classrooms.vercel.app"])
    return app


app = create_app()
mongo = PyMongo(app)


@app.route('/get_chars_image/<_id>', methods=['PUT'])
def get_chars_from_image(_id):
    img = mongo.db.images.find_one({'_id': ObjectId(_id), })

    try:
        text = get_chars(img["url"])
    except:
        response = jsonify({'msg': 'Ocurrió un error con la Imagen: ' + _id})
        response.status_code = 500
        return response

    mongo.db.images.update_one(
        {'_id': ObjectId(_id['$oid']) if '$oid' in _id else ObjectId(_id)},
        {'$set': {'text': text}})

    response = jsonify({'msg': 'Carácteres de la Imagen: ' + _id + ' se han obtenido exitosamente'})
    response.status_code = 200
    return response


if __name__ == "__main__":
    app.run(debug=True, port=3000)
