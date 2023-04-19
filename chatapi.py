import random
import json

import torch
import numpy as np
from model import NeuralNet
from nltk_utils import tokenize, stem





def userChat(userchat):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    
    hidden_size = data["hidden_size"]
    
    output_size = data["output_size"]
    
    tags = data['tags']
    
    model_state = data["model_state"]

    TFIDF_val_vector = data["TFIDF_val_vector"]

    

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    bot_name = "Neural"

    # sentence = "do you use credit cards?"

    

    sentence = userchat

            
    
        
    
    sentence = tokenize(sentence)

   
    # remove duplicates and sort
    ignore_words = ['?', '.', '!', ',']
    sentence_words = [stem(word) for word in sentence if word not in ignore_words]
    sentence_words = sorted(set(sentence_words))

    
    
    # initialize bag with 0 for each word

    New_TFIDF_Vector = np.zeros(len(TFIDF_val_vector), dtype=np.float32)
   
    idx = 0
    for tfidf_word, tfidf_indx in (TFIDF_val_vector):
        
        if tfidf_word in sentence_words:
            New_TFIDF_Vector[idx] = tfidf_indx
        idx +=1


    X = New_TFIDF_Vector
    
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.95:
    
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                return (f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...For more information please contact us Email- id - preeti@nevrio.tech, deepak@nevrio.tech or call us - 9041959799.") 
        return (f"{bot_name}: I do not understand...For more information please contact us Email- id - preeti@nevrio.tech, deepak@nevrio.tech or call us - 9041959799.") 


userChat("How are you")