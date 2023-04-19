from sklearn.feature_extraction.text import TfidfVectorizer
import json
from nltk_utils import tokenize, stem
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)


tags = []
question = []
TFIDF_tags = []

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    
    for pattern in intent['patterns']:
        question.append(pattern)
        # add to tag list
        tags.append(tag)
        TFIDF_tags.append(tag)
        
            

# stem and lower each word

ignore_words = ['?', '.', '!', ',']

# --------------- Preprocessing ---------------------------------

tokenized_list_of_sentences = []
tokenized_list_of_sentence = [tokenize(sentence) for sentence in question]

for each_sentence in tokenized_list_of_sentence:
    tokenized_sentence_list = [stem(word.lower()) for word in each_sentence if word not in ignore_words]
    tokenized_list_of_sentences.append(tokenized_sentence_list)

# ------------------------------------------------


# create training data
X_train = []
y_train = []

     
#--------- TF-IDF ----------------------------
def identity_tokenizer(text):
    return text


v = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)

TFIDF_val_vector = []

transformed_output = v.fit_transform(tokenized_list_of_sentences)

all_feature_names = v.get_feature_names_out()


for word in all_feature_names:
    index = v.vocabulary_.get(word)
    tfidf_index = v.idf_[index]
    TFIDF_val_vector.append((word, tfidf_index))



TFIDF_ques = transformed_output.toarray()

#-----------------------------------------------------------------


TFIDF_tags_pair = zip(TFIDF_ques,TFIDF_tags)

for (TFIDF_ques,TFIDF_tags) in TFIDF_tags_pair:
    TFIDF_ques = np.float32(TFIDF_ques)
    X_train.append(TFIDF_ques)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    
    label = tags.index(TFIDF_tags)
   
    y_train.append(label)
    

#  -------------------------------------------------- 

X_train = np.array(X_train)


y_train = np.array(y_train)



# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, 
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"tags": tags,
"TFIDF_val_vector":  TFIDF_val_vector
}


FILE = "TFIDF_chatbot\data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')