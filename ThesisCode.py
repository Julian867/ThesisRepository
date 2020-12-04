############### Imports ###################
## Initial imports
import pandas as pd
import numpy as np
## Import data
df = pd.read_stata('C:/Users/julia/Desktop/Master/Thesis/data_clear.dta')
pd.set_option('display.max_columns', None)


############### Create target variable #####################
## Create mean moral preference column
df['moral_pref'] = df[['utilitarian', 'insulin_utilitarian']].mean(axis=1)
## Create classification variable
df['target'] = np.where(df['moral_pref'] > 0.5, 1, 0)
## Checking mean of fixations per subject per trial to determine batch size
df_group = df.groupby(['subject', 'trial'])
df_group.size().mean() #21.56 fixations


################ Balancing classes ##################
## See amount of each class
df['target'].value_counts()
## Split df by class
df_class_0, df_class_1 = [x for _, x in df.groupby(df['target'] == 1)]
## Take away from the unbalanced class
df_class_1 = df_class_1.iloc[:39940,:]
## Concatenate classes
classes = [df_class_0, df_class_1]
df = pd.concat(classes)


################ Shuffle and keeping sequences intact ######################
## Groupby function to create groups of each trial per subject
groups = [df for _, df in df.groupby(['subject', 'trial'])]
## Setting seed and shuffling of groups
import random
random.seed(1)
random.shuffle(groups)
## Concatenating them back together and saving in new df
sh_df = pd.concat(groups).reset_index(drop=True)


############### Cleaning ###########################
## Delete columns without integers
sh_df = sh_df.drop(columns = ['subject', 'trial', 'code', 'PersonName', 'GroupName', 'choice', 'utilitarian', 'insulin_utilitarian'])
## Delete columns with NaNs
sh_df = sh_df.dropna(axis='columns')


############# Data groups #################
## Base data grouping: just Eye Tracking data during operations task
df_base = pd.concat([sh_df.iloc[:,18:20], sh_df.iloc[:,29:44]], axis=1, sort=False)
 
## Demographic information
df_dem = pd.concat([sh_df.iloc[:,27:29], sh_df.iloc[:,46]], axis=1, sort=False)
## Surveyed tests scores data
df_score = pd.concat([sh_df.iloc[:,1:7], sh_df.iloc[:,185:193]], axis=1, sort=False)
## Operation task circumstance data
df_circ = pd.concat([sh_df.iloc[:,7], sh_df.iloc[:,9:16], sh_df.iloc[:,44:46]], axis=1, sort=False)
## AOI
df_AOI = pd.concat([sh_df.iloc[:,20:27], sh_df.iloc[:,206:213]], axis=1, sort=False)
## Additional info surveyed tests
df_info = pd.concat([sh_df.iloc[:,107:185], sh_df.iloc[:,195:205]], axis=1, sort=False)
#sh_df.columns.get_loc("dur")

############## Combinations #########################
# Prelimenary 5 + just base
df_base_dem = pd.concat([df_base, df_dem], axis=1, sort=False)
df_base_score = pd.concat([df_base, df_score], axis=1, sort=False)
df_base_circ = pd.concat([df_base, df_circ], axis=1, sort=False)
df_base_AOI = pd.concat([df_base, df_AOI], axis=1, sort=False)
df_all = pd.concat([df_base, df_dem, df_score, df_circ, df_AOI], axis=1, sort=False)
## More advanced groupings:
# Circ and AOI are highly dependent on one another, lets see how they perform
df_base_circ_AOI = pd.concat([df_base, df_circ, df_AOI], axis=1, sort=False)
# See if the poorly performing ones improve base at all
df_base_dem_score_circ = pd.concat([df_base, df_dem, df_score, df_circ], axis=1, sort=False)
# What happens when you remove base?
df_nobase = pd.concat([df_dem, df_score, df_circ, df_AOI], axis=1, sort=False)
# Nobase with 3 least performing groups
df_nobase_dem_score_circ = pd.concat([df_dem, df_score, df_circ], axis=1, sort=False)



############### Creating arrays for preprocessing ######################
## Create new array with classification targets
dependent = sh_df['target'].values
## Create new array with all independent variable data per combination
base_ind = df_base.values
base_dem_ind = df_base_dem.values
base_score_ind = df_base_score.values
base_circ_ind = df_base_circ.values
base_AOI_ind = df_base_AOI.values
base_all_ind = df_all.values
## Advanced groupings testing
base_circ_AOI_ind = df_base_circ_AOI.values
base_dem_score_circ_ind = df_base_dem_score_circ.values
nobase_ind = df_nobase.values
nobase_dem_score_circ_ind = df_nobase_dem_score_circ.values
## Use this to test shapes for traintestsplit column_count input
# nobase_dem_score_circ_ind.shape


################ Splitting #########################
## Import train test split from sklearn
from sklearn.model_selection import train_test_split
## Create function to output training, validation and test and reshaping them
def traintestsplit(independent, column_count):
    x_train,x_test,y_train,y_test = train_test_split(independent,dependent,test_size=0.2,shuffle=False)
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.25,shuffle=False)
    x_train = x_train.reshape(-1, 1, column_count)
    x_val = x_val.reshape(-1, 1, column_count)
    x_test = x_test.reshape(-1, 1, column_count)
    return x_train, y_train, x_val, y_val, x_test, y_test
## Apply function to each of the combinations 
base_x_train, base_y_train, base_x_val, base_y_val, base_x_test, base_y_test = traintestsplit(base_ind, 17)
base_dem_x_train, base_dem_y_train, base_dem_x_val, base_dem_y_val, base_dem_x_test, base_dem_y_test = traintestsplit(base_dem_ind, 20)
base_score_x_train, base_score_y_train, base_score_x_val, base_score_y_val, base_score_x_test, base_score_y_test = traintestsplit(base_score_ind, 31)
base_circ_x_train, base_circ_y_train, base_circ_x_val, base_circ_y_val, base_circ_x_test, base_circ_y_test = traintestsplit(base_circ_ind, 27)
base_AOI_x_train, base_AOI_y_train, base_AOI_x_val, base_AOI_y_val, base_AOI_x_test, base_AOI_y_test = traintestsplit(base_AOI_ind, 31)
base_info_x_train, base_info_y_train, base_info_x_val, base_info_y_val, base_info_x_test, base_info_y_test = traintestsplit(base_info_ind, 119)
base_all_x_train, base_all_y_train, base_all_x_val, base_all_y_val, base_all_x_test, base_all_y_test = traintestsplit(base_all_ind, 59)
## Advanced combinations
base_circ_AOI_x_train, base_circ_AOI_y_train, base_circ_AOI_x_val, base_circ_AOI_y_val, base_circ_AOI_x_test, base_circ_AOI_y_test = traintestsplit(base_circ_AOI_ind, 41)
base_dem_score_circ_x_train, base_dem_score_circ_y_train, base_dem_score_circ_x_val, base_dem_score_circ_y_val, base_dem_score_circ_x_test, base_dem_score_circ_y_test = traintestsplit(base_dem_score_circ_ind, 44)
nobase_x_train, nobase_y_train, nobase_x_val, nobase_y_val, nobase_x_test, nobase_y_test = traintestsplit(nobase_ind, 41)
nobase_dem_score_circ_x_train, nobase_dem_score_circ_y_train, nobase_dem_score_circ_x_val, nobase_dem_score_circ_y_val, nobase_dem_score_circ_x_test, nobase_dem_score_circ_y_test = traintestsplit(nobase_dem_score_circ_ind, 27)


################ Preprocessing ###############################
## Transform data
#from sklearn.preprocessing import StandardScaler
#stndrd=StandardScaler()
#x_train=stndrd.fit_transform(x_train)
#x_val=stndrd.transform(x_val)
#x_test=stndrd.transform(x_test)
## Reshaping for input into model (adding third dimension)
#x_train = x_train.reshape(-1, 1, last_column)
#x_val = x_val.reshape(-1, 1, last_column)
#x_test = x_test.reshape(-1, 1, last_column)


################# Simple RNN ###########################
## Import for model
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout
## Define optimizer
opt = keras.optimizers.Adam(learning_rate=0.000001)
## A function for the RNN model
def buildRNNmodel():
    RNN_model = Sequential()
    RNN_model.add(SimpleRNN(300, return_sequences=True))
    RNN_model.add(Dropout(0.2))
    RNN_model.add(SimpleRNN(150, return_sequences=True))
    RNN_model.add(Dropout(0.2))
    RNN_model.add(Dense(1, activation='sigmoid'))
    RNN_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return RNN_model
## Initialize each RNN model
RNN_model = buildRNNmodel()
RNN_model1 = buildRNNmodel()
RNN_model2 = buildRNNmodel()
RNN_model3 = buildRNNmodel()
RNN_model4 = buildRNNmodel()
RNN_model5 = buildRNNmodel()
RNN_model_a = buildRNNmodel()
RNN_model_a1 = buildRNNmodel()
RNN_model_a2 = buildRNNmodel()
RNN_model_a3 = buildRNNmodel()


################ LSTM ##########################
## Import for model
from keras.layers import LSTM
## Build the model
def buildLSTMmodel():
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(300, return_sequences=True))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(LSTM(150, return_sequences=True))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(Dense(1, activation='sigmoid'))
    LSTM_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return LSTM_model
## Initialize each LSTM model
LSTM_model = buildLSTMmodel()
LSTM_model1 = buildLSTMmodel()
LSTM_model2 = buildLSTMmodel()
LSTM_model3 = buildLSTMmodel()
LSTM_model4 = buildLSTMmodel()
LSTM_model5 = buildLSTMmodel()
LSTM_model_a = buildLSTMmodel()
LSTM_model_a1 = buildLSTMmodel()
LSTM_model_a2 = buildLSTMmodel()
LSTM_model_a3 = buildLSTMmodel()



############### Fitting the models #################
## Histories RNN
base_RNN_history = RNN_model.fit(base_x_train, base_y_train, validation_data=(base_x_val, base_y_val), epochs=100, batch_size=20)
base_dem_RNN_history = RNN_model1.fit(base_dem_x_train, base_dem_y_train, validation_data=(base_dem_x_val, base_dem_y_val), epochs=100, batch_size=20)
base_score_RNN_history = RNN_model2.fit(base_score_x_train, base_score_y_train, validation_data=(base_score_x_val, base_score_y_val), epochs=100, batch_size=20)
base_circ_RNN_history = RNN_model3.fit(base_circ_x_train, base_circ_y_train, validation_data=(base_circ_x_val, base_circ_y_val), epochs=100, batch_size=20)
base_AOI_RNN_history = RNN_model4.fit(base_AOI_x_train, base_AOI_y_train, validation_data=(base_AOI_x_val, base_AOI_y_val), epochs=100, batch_size=20)
base_all_RNN_history = RNN_model5.fit(base_all_x_train, base_all_y_train, validation_data=(base_all_x_val, base_all_y_val), epochs=100, batch_size=20)
## Advanced groupings histories RNN
base_circ_AOI_RNN_history = RNN_model_a.fit(base_circ_AOI_x_train, base_circ_AOI_y_train, validation_data=(base_circ_AOI_x_val, base_circ_AOI_y_val), epochs=100, batch_size=20)
base_dem_score_circ_RNN_history = RNN_model_a1.fit(base_dem_score_circ_x_train, base_dem_score_circ_y_train, validation_data=(base_dem_score_circ_x_val, base_dem_score_circ_y_val), epochs=100, batch_size=20)
nobase_RNN_history = RNN_model_a2.fit(nobase_x_train, nobase_y_train, validation_data=(nobase_x_val, nobase_y_val), epochs=100, batch_size=20)
nobase_dem_score_circ_RNN_history = RNN_model_a3.fit(nobase_dem_score_circ_x_train, nobase_dem_score_circ_y_train, validation_data=(nobase_dem_score_circ_x_val, nobase_dem_score_circ_y_val), epochs=100, batch_size=20)
## Histories LSTM
base_LSTM_history = LSTM_model.fit(base_x_train, base_y_train, validation_data=(base_x_val, base_y_val), epochs=100, batch_size=20)
base_dem_LSTM_history = LSTM_model1.fit(base_dem_x_train, base_dem_y_train, validation_data=(base_dem_x_val, base_dem_y_val), epochs=100, batch_size=20)
base_score_LSTM_history = LSTM_model2.fit(base_score_x_train, base_score_y_train, validation_data=(base_score_x_val, base_score_y_val), epochs=100, batch_size=20)
base_circ_LSTM_history = LSTM_model3.fit(base_circ_x_train, base_circ_y_train, validation_data=(base_circ_x_val, base_circ_y_val), epochs=100, batch_size=20)
base_AOI_LSTM_history = LSTM_model4.fit(base_AOI_x_train, base_AOI_y_train, validation_data=(base_AOI_x_val, base_AOI_y_val), epochs=100, batch_size=20)
base_all_LSTM_history = LSTM_model5.fit(base_all_x_train, base_all_y_train, validation_data=(base_all_x_val, base_all_y_val), epochs=100, batch_size=20)
## Advanced groupings histories LSTM
base_circ_AOI_LSTM_history = LSTM_model_a.fit(base_circ_AOI_x_train, base_circ_AOI_y_train, validation_data=(base_circ_AOI_x_val, base_circ_AOI_y_val), epochs=100, batch_size=20)
base_dem_score_circ_LSTM_history = LSTM_model_a1.fit(base_dem_score_circ_x_train, base_dem_score_circ_y_train, validation_data=(base_dem_score_circ_x_val, base_dem_score_circ_y_val), epochs=100, batch_size=20)
nobase_LSTM_history = LSTM_model_a2.fit(nobase_x_train, nobase_y_train, validation_data=(nobase_x_val, nobase_y_val), epochs=100, batch_size=20)
nobase_dem_score_circ_LSTM_history = LSTM_model_a3.fit(nobase_dem_score_circ_x_train, nobase_dem_score_circ_y_train, validation_data=(nobase_dem_score_circ_x_val, nobase_dem_score_circ_y_val), epochs=100, batch_size=20)



############### Visualization ############
## Comparing histories function
import matplotlib.pyplot as plt
def compare_visualize(history1,history2):
    # summarize history for accuracy
    plt.plot(history1.history['accuracy'])
    plt.plot(history1.history['val_accuracy'])
    plt.plot(history2.history['accuracy'])
    plt.plot(history2.history['val_accuracy'])
    plt.title('Compared model accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0.40, 1.00)
    plt.xlabel('Epoch')
    plt.legend(['RNN train', 'RNN val', 'LSTM train', 'LSTM val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history1.history['loss'])
    plt.plot(history1.history['val_loss'])
    plt.plot(history2.history['loss'])
    plt.plot(history2.history['val_loss'])
    plt.title('Compared model loss')
    plt.ylabel('Loss')
    plt.ylim(0.00, 1.00)
    plt.xlabel('Epoch')
    plt.legend(['RNN train', 'RNN val', 'LSTM train', 'LSTM val'], loc='lower left')
    return plt.show()
## Comparing histories
print("COMPARED HISTORIES:")
print("base history:")
compare_visualize(base_RNN_history, base_LSTM_history)
print("base + dem history:")
compare_visualize(base_dem_RNN_history, base_dem_LSTM_history)
print("base + score history:")
compare_visualize(base_score_RNN_history, base_score_LSTM_history)
print("base + circ history:")
compare_visualize(base_circ_RNN_history, base_circ_LSTM_history)
print("base + AOI history:")
compare_visualize(base_AOI_RNN_history, base_AOI_LSTM_history)
print("base + all history:")
compare_visualize(base_all_RNN_history, base_all_LSTM_history)
## Computing advanced groupings histories
print("base + circ + AOI history:")
compare_visualize(base_circ_AOI_RNN_history, base_circ_AOI_LSTM_history)
print("base + dem + score + circ history:")
compare_visualize(base_dem_score_circ_RNN_history, base_dem_score_circ_LSTM_history)
print("nobase history:")
compare_visualize(nobase_RNN_history, nobase_LSTM_history)
print("dem + score + circ history:")
compare_visualize(nobase_dem_score_circ_RNN_history, nobase_dem_score_circ_LSTM_history)



############### Evaluation #################
## Single model evaluation
RNN_scores = RNN_model_a3.evaluate(nobase_dem_score_circ_x_test, nobase_dem_score_circ_y_test, verbose=0)
print("Accuracy RNN: %.2f%%" % (RNN_scores[1]*100))
print("Loss RNN: ", RNN_scores[0])
LSTM_scores = LSTM_model_a3.evaluate(nobase_dem_score_circ_x_test, nobase_dem_score_circ_y_test, verbose=0)
print("Accuracy LSTM: %.2f%%" % (LSTM_scores[1]*100))
print("Loss LSTM: ", LSTM_scores[0])
## Evaluation of each RNN model on test sets
base_RNN_score = RNN_model.evaluate(base_x_test, base_y_test, verbose=0)
base_dem_RNN_score = RNN_model1.evaluate(base_dem_x_test, base_dem_y_test, verbose=0)
base_score_RNN_score = RNN_model2.evaluate(base_score_x_test, base_score_y_test, verbose=0)
base_circ_RNN_score = RNN_model3.evaluate(base_circ_x_test, base_circ_y_test, verbose=0)
base_AOI_RNN_score = RNN_model4.evaluate(base_AOI_x_test, base_AOI_y_test, verbose=0)
base_all_RNN_score = RNN_model5.evaluate(base_all_x_test, base_all_y_test, verbose=0)
## Evaluation of each LSTM model on test sets
base_LSTM_score = LSTM_model.evaluate(base_x_test, base_y_test, verbose=0)
base_dem_LSTM_score = LSTM_model1.evaluate(base_dem_x_test, base_dem_y_test, verbose=0)
base_score_LSTM_score = LSTM_model2.evaluate(base_score_x_test, base_score_y_test, verbose=0)
base_circ_LSTM_score = LSTM_model3.evaluate(base_circ_x_test, base_circ_y_test, verbose=0)
base_AOI_LSTM_score = LSTM_model4.evaluate(base_AOI_x_test, base_AOI_y_test, verbose=0)
base_all_LSTM_score = LSTM_model5.evaluate(base_all_x_test, base_all_y_test, verbose=0)
## Create bar chart comparing all evaluation scores
import matplotlib
labels = ['Base (B)', 'B + dem', 'B + score', 'B + circ', 'B + AOI', 'All']
RNN_scores = [base_RNN_score[1]*100, base_dvem_LSTM_score[1]*100, base_score_RNN_score[1]*100, base_circ_RNN_score[1]*100, base_AOI_RNN_score[1]*100, base_all_RNN_score[1]*100]
LSTM_scores = [base_LSTM_score[1]*100, base_dem_LSTM_score[1]*100, base_score_LSTM_score[1]*100, base_circ_LSTM_score[1]*100, base_AOI_LSTM_score[1]*100, base_all_LSTM_score[1]*100]
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, RNN_scores, width, label='RNN')
rects2 = ax.bar(x + width/2, LSTM_scores, width, label='LSTM')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 100)
ax.set_title('Evaluation on test sets per model by data grouping')
ax.set_yticks(np.arange(0,100, step=10))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()