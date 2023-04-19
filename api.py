# Using flask to make an api
# import necessary libraries and functions
from flask import Flask, jsonify, request
from chatapi import userChat
  
# creating a Flask app
app = Flask(__name__)
  
# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/', methods = ['GET', 'POST'])
def home():
    if(request.method == 'GET'):
  
        data = userChat(request.args['input'])
        return jsonify({'data':data})

    if(request.method == 'POST'):

        data = userChat(request.json['question'])
        
        
        print(jsonify(data))
        return jsonify({'data': data})

  
  
if __name__ == '__main__':
  
    app.run(debug = True)
