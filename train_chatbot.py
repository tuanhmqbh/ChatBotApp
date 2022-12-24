import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']

#Load dữ liệu từ file intents.json, đây là file chứa các mẫu câu giao tiếp và phản hồi được định nghĩa trước
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Tiền xử lý dữ liệu và mã hóa 
for intent in intents['intents']:
    for pattern in intent['patterns']:

        #Mã hóa từng từ một
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #Nối từng từ vào danh sách từ (kho tài liệu)
        documents.append((w, intent['tag']))

        # Thêm vào danh sách lớp các thẻ
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Thực hiện bổ ngữ, chuyển về viết thường từng từ và loại bỏ trùng lắp
# Ví dụ letomaze (bổ ngữ): lower, lowest, low đều đưa về từ gốc là low
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sắp xếp các lớp
classes = sorted(list(set(classes)))
# Document là sự kết hợp giữa mẫu và các ý định (trả lời)
print (len(documents), "documents")
# classes là các ý định
# words = tất cả các từ và từ vựng
print (len(words), "unique lemmatized words", words)

#Tạo các file pkl để lưu trữ các đối tượng dùng để dự báo câu trả lời
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# Tạo dữ liệu huấn luyện
training = []
# Tạo một mảng rỗng chứa dữ liệu đầu ra
output_empty = [0] * len(classes)
# Tập huấn luyện, "túi đựng từ" cho từng câu
for doc in documents:
    # Khởi tạo bag chưa các từ
    bag = []
    # Danh sách các từ được mã hóa cho mẫu
    pattern_words = doc[0]
    # Tạo "bổ ngữ" cho các từ: Tạo từ cơ bản và biểu diễn được cho các từ liên quan
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Nếu từ được tìm thấy trong mẫu thì them 1 vào túi bag
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Với mỗi mẫu, gán 0 cho các thẻ, 1 cho thẻ hiện tại
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# Trộn các tính năng và lưu vào mảng np
random.shuffle(training)
training = np.array(training)
# Tạo danh sách huấn luyện và thử nghiệm. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Tạo mô hình mạng nơ-ron sâu 3 lớp sử dụng API của Keras: Lớp 1 có 128 neurons, Lớp 2 có 64 neurons 
# và lớp 3 là lớp output bao gồm số nơ-ron bằng số ý định để dự đoán
#Lưu mô hình vào file chatbot_model.h5
model = Sequential()

#Sử dụng hàm relu để kích hoạt trong Neural Network
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#Hàm trung bình mũ softmax để tính xác suất
model.add(Dense(len(train_y[0]), activation='softmax'))

# Biên dịch mô hình
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Tính sự phù hợp và lưu mô hình 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")
