import gensim
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from trnlp import SpellingCorrector
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from tensorflow.python.keras.models import Sequential, model_from_json
from tensorflow.python.keras.layers import Dense, GRU
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from sklearn.model_selection import KFold


def calculate_distance(current_word, possibility_1, possibility_2):
    minimum_length = min(len(current_word), len(possibility_1), len(possibility_2))

    poss_1_distance = 0
    poss_2_distance = 0

    for c in range(minimum_length):

        try:
            if current_word[c] != possibility_1[c]:
                f_point = keyboard[current_word[c]][possibility_1[c]]
                poss_1_distance += f_point
            poss_1_distance += 0
        except KeyError:
            poss_1_distance += 100
        try:
            if current_word[c] != possibility_2[c]:
                f_point = keyboard[current_word[c]][possibility_2[c]]
                poss_2_distance += f_point
            poss_2_distance += 0
        except KeyError:
            poss_2_distance += 100

    if poss_2_distance < poss_1_distance:
        return 1
    else:
        return 0


def com2vec(f_model, f_x, f_y):
    f_x_train = list()
    f_y_train = list()

    temp_vec = list()

    for q in range(WORD2VEC_VECTOR_SIZE):
        temp_vec.append(0)

    f_index = 0

    at_least_one = False
    word_count = 0

    for f_com in f_x:

        for f_word in f_com:
            try:
                com_vec = f_model.wv.get_vector(f_word)
                temp_vec = [temp_vec[p] + com_vec[p] for p in range(len(com_vec))]
                at_least_one = True
                word_count += 1
            except KeyError:
                pass
        if at_least_one:
            temp_vec = [number / word_count for number in temp_vec]
            f_to_be_added = list(temp_vec)
            f_x_train.append((f_to_be_added, f_index))
            f_y_train.append(f_y[f_index])
        word_count = 0
        at_least_one = False
        f_index += 1
        for q in range(WORD2VEC_VECTOR_SIZE):
            temp_vec[q] = 0

    return f_x_train, f_y_train


def one_com2vec(load_model):
    f_temp_comment = ""
    com = input("Please enter a comment: ")
    com_tokens = tokenizer.tokenize(com)
    for tok in com_tokens:
        tok = tok.lower()
        if tok not in stopwords.words('turkish'):
            s_corrector.settext(tok)
            f_corrected = s_corrector.correction(unrepeater=True, transposes=True, insert=True, replaces=True)
            if not (len(f_corrected[0]) < 2):
                f_index = calculate_distance(tok.lower(), f_corrected[0][0].lower(), f_corrected[0][1].lower())
            else:
                f_index = 0
            f_to_be_added = f_corrected[0][f_index].lower()
            f_temp_comment += f_to_be_added + " "

    f_pre_comment = f_temp_comment.strip().split()

    temp_f_x = list()
    temp_f_y = list()
    temp_f_x.append(f_pre_comment)
    temp_f_y.append(0)

    f_x_res, f_y_res = com2vec(load_model, temp_f_x, temp_f_y)

    return f_x_res[0][0]


WORD2VEC_VECTOR_SIZE = 64
WORD2VEC_WINDOW_SIZE = 3
WORD2VEC_MINWORD_COUNT = 10
EPOCH = 5
BATCH_SIZE = 1024

positive_comment_count = 0
negative_comment_count = 0
count = 0
COMMENT_PERCENTAGE = 0.90

comment_points = list()
positive_comments = list()
negative_comments = list()

tokenizer = RegexpTokenizer("[\\w']+")

s_corrector = SpellingCorrector()

keyboard = dict()

with open('Comments.txt', 'r') as f:
    comments = f.readlines()

with open('Keyboard.txt', 'r') as f:
    board = f.readlines()

for line in board:
    chars = line.split()
    keyboard[chars[0]] = {}
    for i in range(1, len(chars)):
        keyboard[chars[0]][chars[i]] = 1
"""
for comment in comments:
    comment_flag = -1
    temp_comment = ""
    tokens = tokenizer.tokenize(comment)
    point = (int(tokens.pop(-1)) + int(tokens.pop(-1)) + int(tokens.pop(-1))) / 3

    if point >= 7:
        comment_flag = 1
        positive_comment_count += 1
        comment_points.append(1)
    else:
        comment_flag = 0
        negative_comment_count += 1
        comment_points.append(0)

    for token in tokens:
        token = token.lower()
        if token not in stopwords.words('turkish'):
            s_corrector.settext(token)
            corrected = s_corrector.correction(unrepeater=True, transposes=True, insert=True, replaces=True)
            if not (len(corrected[0]) < 2):
                index = calculate_distance(token.lower(), corrected[0][0].lower(), corrected[0][1].lower())
            else:
                index = 0
            to_be_added = corrected[0][index].lower()
            temp_comment += to_be_added + " "

    pre_comment = temp_comment.strip().split()
    if comment_flag == 1:
        positive_comments.append(pre_comment)
    else:
        negative_comments.append(pre_comment)
    count += 1


print(
    "Positive comments: {} {}".format(positive_comment_count * 100 / (positive_comment_count + negative_comment_count),
                                      positive_comment_count))
print(
    "Positive comments: {} {}".format(negative_comment_count * 100 / (positive_comment_count + negative_comment_count),
                                      negative_comment_count))

cut_off = int((len(negative_comments) * COMMENT_PERCENTAGE))  # cutt_off değeri negatif

p_comments = positive_comments[:len(negative_comments)]

train_comment_list = p_comments[:cut_off] + negative_comments[:cut_off]
train_point_list = [1 for x in range(len(p_comments[:cut_off]))] + [0 for x in range(len(negative_comments[:cut_off]))]


test_comment_list = p_comments[cut_off:] + negative_comments[cut_off:]
test_point_list = [1 for x in range(len(p_comments[cut_off:]))] + [0 for x in range(len(negative_comments[cut_off:]))]

print("\nTotal Comments\nPositive: {}\nNegative: {}\nTotal: {}".format(len(p_comments), len(negative_comments),
                                                                       len(p_comments) + len(negative_comments)))
print("\nTrain Percentage: %", COMMENT_PERCENTAGE * 100)
print("\nTrain Comment Count\nPositive: {}\nNegative: {}\nTotal: {}".format(len(p_comments[:cut_off]),
                                                                            len(negative_comments[:cut_off]),
                                                                            len(train_comment_list)))

print("\nTest Comment Count\nPositive: {}\nNegative: {}\nTotal: {}".format(len(p_comments[cut_off:]),
                                                                           len(negative_comments[cut_off:]),
                                                                           len(test_comment_list)))

model = Word2Vec(train_comment_list, size=WORD2VEC_VECTOR_SIZE, window=WORD2VEC_WINDOW_SIZE,
                 min_count=WORD2VEC_MINWORD_COUNT, sg=1)

model.save("word2vec.model")
print("Saved word2vec.")

x, y = com2vec(model, train_comment_list, train_point_list)
x1, y1 = com2vec(model, test_comment_list, test_point_list)


f = open("TrainDataset.txt", "w")
for j in range(len(x)):
    line = str(x[j][0])[1:-2].replace(" ", "") + "," + str(y[j]) + "\n"
    f.write(line)
f.close()

f = open("TestDataset.txt", "w")
for j in range(len(x1)):
    line = str(x1[j][0])[1:-2].replace(" ", "") + "," + str(y1[j]) + "\n"
    f.write(line)
f.close()

print("\nTrain Data Set Comment Count: {}\nEmpty Comment Count: {} \nTotal Comment Count: {}"
      .format(len(x), len(train_comment_list) - len(x), len(train_comment_list)))

print("\nTest Data Set Comment Count: {}\nEmpty Comment Count: {} \nTotal Comment Count: {}"
      .format(len(x1), len(test_comment_list) - len(x1), len(test_comment_list)))
#######################################################################################################################

temp_train = list()
temp_test = list()

for first, second in x:
    temp_train.append(first)
train = np.array(temp_train)
train_point = np.array(y)

train = train.reshape(*train.shape, 1)


for first, second in x1:
    temp_test.append(first)
test = np.array(temp_test)
test_point = np.array(y1)


test = test.reshape(*test.shape, 1)


sm_model = Sequential()
sm_model.add(Dense(WORD2VEC_VECTOR_SIZE, input_shape=train.shape[1:]))
sm_model.add(GRU(units=64, return_sequences=True, recurrent_activation="sigmoid"))
sm_model.add(GRU(units=16, return_sequences=True, recurrent_activation="sigmoid"))
sm_model.add(GRU(units=8, return_sequences=True, recurrent_activation="sigmoid"))
sm_model.add(GRU(units=4, recurrent_activation="sigmoid"))
sm_model.add(Dense(1, activation="sigmoid"))

optimizer = Adam(lr=1e-3)

sm_model.compile(loss='binary_crossentropy',
                 optimizer=optimizer,
                 metrics=['accuracy'])


sm_model.summary()

sm_model.fit(train, train_point, epochs=EPOCH, batch_size=BATCH_SIZE)

print("---TESTING---")
result = sm_model.evaluate(test, test_point)

print("---DONE---")


predictions = sm_model.predict(test)
precision, recall, fscore, support = metrics.precision_recall_fscore_support(test_point, np.round(predictions),
                                                                             zero_division=False)
print("Accuracy: ", result[1])
print('Precision: {}'.format(precision[0]))
print('Recall: {}'.format(recall[0]))
print('F-Score: {}'.format(fscore[0]))
print('ROC:', metrics.roc_auc_score(test_point, np.round(predictions)))
print('MSE:', metrics.mean_squared_error(test_point, np.round(predictions)))
print('RMSE:', metrics.mean_squared_error(test_point, np.round(predictions), squared=False))
print('MAE:', metrics.mean_absolute_error(test_point, np.round(predictions)))
print('R-Squared:', metrics.r2_score(test_point, np.round(predictions)))
print("--------Cross Validation---------")


folds = KFold(n_splits=5, shuffle=True, random_state=35)
scores = []
for n_fold, (train_index, valid_index) in enumerate(folds.split(train, train_point)):
    print('\n Fold ' + str(n_fold + 1))

    X_train, X_test = train[train_index], train[valid_index]
    y_train, y_test = train_point[train_index], train_point[valid_index]

    sm_model.fit(X_train, y_train, batch_size=BATCH_SIZE)
    y_pred = sm_model.predict(X_test)

    acc_score = metrics.accuracy_score(y_test, np.round(y_pred))
    scores.append(acc_score)
    print('\n Accuracy score for Fold ' + str(n_fold + 1) + ' --> ' + str(acc_score) + '\n')

print('Avg. accuracy score of CV:' + str(np.mean(scores)))

model_json = sm_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

sm_model.save_weights("model.h5")
print("Saved model to disk")
"""
#################################### MODEL LOAD PART########################################

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.summary()

new_model = Word2Vec.load("word2vec.model")

ilker = 0
while ilker < 5:
    test_comment = one_com2vec(new_model)
    f = open("TestComment.txt", "w")
    line = str(test_comment)[1:-2].replace(" ", "") + "\n"
    f.write(line)
    f.close()
    kk = list()
    kk.append(test_comment)
    qw = np.array(kk)
    qw = qw.reshape(*qw.shape, 1)
    pred = loaded_model.predict(qw)
    ilker += 1
    print(pred)
