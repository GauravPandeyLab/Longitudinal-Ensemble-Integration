from sklearn.model_selection import train_test_split
from eipy.metrics import fmax_score
import pandas as pd
import numpy as np
import eipy.ei as e
import tensorflow as tf
from keras.callbacks import EarlyStopping # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import LSTM,Dense, Bidirectional, GRU, Dropout, SimpleRNN, BatchNormalization # type: ignore
from keras.regularizers import l2 # type: ignore
import matplotlib.pyplot as plt
from keras import losses

def ohe(y):
    labels_across_time = np.eye(3)[y]

    return labels_across_time

def initial_data_label_prep(data_nested_dict, label_dict):
    data_list_of_dicts = [data_nested_dict[k] for k in data_nested_dict.keys()]
    
    for k,v in label_dict.items():
        label_dict[k] = v.reset_index(drop=True)
    labels = pd.DataFrame(label_dict)
    labels = labels.to_numpy()
    encoding_dict = {'NL': 0, 'MCI': 1, 'Dementia': 2}
    labels = np.vectorize(lambda x: encoding_dict[x])(labels)

    return data_list_of_dicts, labels

def get_column_names(df):
    column_names = []
    for i in range(df.columns.nlevels):
        if i == 0:
            column_names.append(df.columns.get_level_values(i).unique().drop("labels"))
            
        else:
            column_names.append(df.columns.get_level_values(i).unique().drop(''))
    
    return column_names

def fix_first_time_point(df):
    new_columns = get_column_names(df)
    classes=[0,1,2]
    new_columns.append(classes)
    new_mux=pd.MultiIndex.from_product(iterables=new_columns, names=["modality", "base predictor", "sample", "class"])
    new_df = pd.DataFrame(columns=new_mux)

    for col in new_df.columns:
        if col[-1] == 0:
            new_df[col] = 1 - df[col[:-1]]
        elif col[-1] == 1:
            new_df[col] = df[col[:-1]]
        else:
            new_df[col] = 0
    
    new_df['labels'] = df['labels']

    return new_df

def generate_RNN_data(seed, metrics, base_predictors, data, y, aligned=True):
    meta_data = []
    for t in range(len(data)):
        #time dependent data splitting
        X_train_test_timestep = data[t]
        labels_at_timestep = y[:, t]
        EI_for_timestep = e.EnsembleIntegration(
                            base_predictors=base_predictors,
                            k_outer=5,
                            k_inner=5,
                            n_samples=1,
                            sampling_strategy=None,
                            sampling_aggregation="mean",
                            n_jobs=-1,
                            metrics=metrics,
                            random_state=seed,
                            project_name=f"time step {t}",
                            model_building=False,
                            time_series=True
                            )
        print(f"generating metadata for timestep {t}")
        EI_for_timestep.fit_base(X_train_test_timestep, labels_at_timestep)
        meta_data.append([EI_for_timestep.ensemble_training_data, EI_for_timestep.ensemble_test_data, EI_for_timestep.base_summary])

    #swap arrangement across folds and time
    RNN_training_data = [[dfs[0][i] for dfs in meta_data] for i in range(5)]
    RNN_test_data = [[dfs[1][i] for dfs in meta_data] for i in range(5)]
    base_summaries = [x[-1] for x in meta_data]

    if aligned: # if data and label are aligned
        for i in range(len(RNN_training_data)):
            RNN_training_data[i][0] = fix_first_time_point(RNN_training_data[i][0])
            RNN_test_data[i][0] = fix_first_time_point(RNN_test_data[i][0])
    
    return RNN_training_data, RNN_test_data, base_summaries



def reformat_data(dfs):
    #reformat labels
    labels_across_time = np.column_stack([df['labels'].values for df in dfs])
    labels_across_time = np.eye(3)[labels_across_time]
    
    # reformat data
    RNN_training_data_fold = [df.drop(columns=["labels"], axis=1, level=0) for df in dfs]
    data_arrays_per_timepoint = [df.to_numpy() for df in RNN_training_data_fold]
    tensor_3d = np.stack(data_arrays_per_timepoint, axis=1)

    return tensor_3d, labels_across_time



def build_RNN(units, cell, drout, L2, deep=True, reg_layer='batchnorm', activation='tanh', gamma=0):
    model = Sequential()
    
    params_dict_1 = {'units': units,
                     'activity_regularizer': l2(L2),
                     'dropout': drout,
                     'recurrent_dropout': drout,
                     'return_sequences': True}
    params_dict_2 = {'units': units // 2,
                     'activity_regularizer': l2(L2),
                     'dropout': drout,
                     'recurrent_dropout': drout,
                     'return_sequences': True}
    

    if cell =='RNN':
        model.add(SimpleRNN(**params_dict_1))
    elif cell == 'biGRU':
        model.add(Bidirectional(GRU(**params_dict_1)))
    elif cell == 'biLSTM':
        model.add(Bidirectional(LSTM(**params_dict_1)))
    elif cell == 'GRU':
        model.add(GRU(**params_dict_1))
    elif cell == 'LSTM':
        model.add(LSTM(**params_dict_1))
    
    
    if deep:
        model.add(Dropout(drout))
        if reg_layer=="batchnorm":
            model.add(BatchNormalization())
        
        if cell == 'RNN':
            model.add(SimpleRNN(**params_dict_2))
        if cell == 'biGRU':
            model.add(Bidirectional(GRU(**params_dict_2)))
        elif cell == 'biLSTM':
            model.add(Bidirectional(LSTM(**params_dict_2)))
        elif cell == 'GRU':
            model.add(GRU(**params_dict_2))
        elif cell == 'LSTM':
            model.add(LSTM(**params_dict_2))
        
    # model.add(Dropout(drout))
    # if reg_layer=="batchnorm":
    #     model.add(BatchNormalization())
    
    # MLP Classification model    
    model.add(Dense(units // 2 , activation=activation))
    model.add(Dropout(drout))
    if reg_layer=="batchnorm":
        model.add(BatchNormalization())        
    
    model.add(Dense(units // 4  , activation=activation))
    model.add(Dropout(drout))
    if reg_layer=="batchnorm":
        model.add(BatchNormalization())

    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss=occ_loss(gamma=gamma), optimizer='adam')
    return model

# find the fmax of every class
def fmax_ova(y_test, y_pred, beta=1.0):
    fmaxes = []
    for i in range(len(np.unique(y_test))): # number of unique classes
        y_test_class = np.where(y_test == i, 1, 0)
        y_pred_class = y_pred[:, i]

        fmaxes.append(fmax_score(y_test_class, y_pred_class, beta=beta))
    return fmaxes        

#use data up to time point n, 
#e.g., when n=3 means use data from t=0,1,2,3 and predict on label from t=n+1=4
cell_list = ["LSTM", "GRU", "biLSTM"]
f_scores = {k: [] for k in cell_list}
def train_eval_RNNs(RNN_training_data, RNN_test_data, n=3, cells=cell_list):
    global f_scores
    for cell in cells:
        y_preds = []
        y_tests = []
        for i, dfs in enumerate(RNN_training_data):
            print(f"{cell}, fold {i+1}")
            
            #numpy-ify data and slice at timepoint n
            X_train, y_train = reformat_data(dfs)
            X_train, y_train = X_train[:,:n,:], y_train[:,n,:]
            X_test, y_test = reformat_data(RNN_test_data[i])
            X_test, y_test = X_test[:,:n,:], y_test[:,n,:]

            model = build_RNN(units=64, cell=cell, drout=0.2, L2=0.00) #changed from 32 units to 64
            
            #create val set and implement early stopping
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, stratify=y_train, test_size=0.1, random_state=i**2)
            early_stop = EarlyStopping(
                monitor='val_loss', patience=20, verbose=1,
                restore_best_weights=True, start_from_epoch=15)
            
            #fit model
            model.fit(
                X_train, y_train, epochs=2000, validation_data=[X_val, y_val],
                verbose=1, callbacks=[early_stop])
            
            '''for fixed number of epochs'''
            # #epochs were found by recording the best epoch over 5 folds over 5 runs
            # if cell == "LSTM" or "GRU":
            #     model.fit(X_train, y_train, epochs=20, validation_data=[X_test, y_test], verbose=1)
            # elif cell =='biLSTM':
            #     model.fit(X_train, y_train, epochs=18, validation_data=[X_test, y_test], verbose=1)


            y_pred = model.predict(X_test)
            y_preds.append(y_pred)
            
            y_test = np.array([np.argmax(ohe_label, axis=-1) for ohe_label in y_test]) # convert back to ordinal
            y_tests.append(y_test)

        #turn into single lists
        y_preds = np.concatenate(y_preds, axis=0)
        y_tests = np.concatenate(y_tests, axis=0)
        
        f_scores[cell].append(fmax_ova(y_tests, y_preds, beta=1))
        print(f_scores)

def remove_thresholds_from_dict(scores_dict):
    return {k : np.array([[x[0] for x in scores_dict[k][i]] for i in range(len(scores_dict[k]))]) for k in scores_dict.keys()}

def build_time_plot(list_of_scores_dicts, cell):
    list_of_scores_dicts_no_threshold = remove_thresholds_from_dict(list_of_scores_dicts)
    class_names = ["CN", "MCI", "AD"]
    # Calculate median values for each class at every time point
    f_scores_up_to_t_no_threshold_cell = np.array([t_dict[cell] for t_dict in list_of_scores_dicts_no_threshold])
    medians = np.median(f_scores_up_to_t_no_threshold_cell, axis=1)

    # Create line graphs for each class
    plt.figure(figsize=(10, 6))  # Set the figure size

    for idx, class_name in enumerate(class_names):
        # Plot median values for each class
        plt.plot(medians[:, idx], label=class_name)
        # Annotate the plot with median values
        for i, median_val in enumerate(medians[:, idx]):
            plt.text(i, median_val, f"{median_val:.2f}", ha='center', va='bottom', color='black')


        # # Add box plots for the full data set (5 samples) at each time point
        # box_data = f_scores_up_to_t_no_threshold_cell[:, :, idx]
        # positions = np.arange(0, box_data.shape[0]) + idx * 0.2  # Adjust positions for box plots
        # plt.boxplot(box_data, positions=positions, widths=0.15)

    plt.xlabel("data up to time=n with labels at time=n+1")  # Set the x-axis label
    plt.ylabel("median fmax")  # Set the y-axis label
    plt.title(f"median class fmax over time with {cell}")  # Set the title
    plt.legend()  # Show legend with class names

    # Customize x-axis ticks
    num_time_points = medians.shape[0]
    plt.xticks(range(num_time_points), range(1, num_time_points + 1))  # Set x ticks as integers

    # Add circles at integer time points
    plt.plot(range(num_time_points), medians[:, 0], 'o', color='tab:blue')  # Circles for CN
    plt.plot(range(num_time_points), medians[:, 1], 'o', color='tab:orange')  # Circles for MCI
    plt.plot(range(num_time_points), medians[:, 2], 'o', color='tab:green')  # Circles for AD


    plt.show()  # Display the plot

def build_boxplots_for_timepoint(list_of_scores_dicts, n, cell):
    data_dict = list_of_scores_dicts[n]
    # Define colors for each group of 3 boxes
    colors = ['blue', 'orange', 'green']
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))


    for i, (key, value) in enumerate(data_dict.items()):
        x_positions = np.arange(value.shape[1]) + i * 4  # Adjust spacing between groups of boxes
        for j in range(value.shape[1]):
            ax.boxplot(value[:, j], positions=[x_positions[j]], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=colors[j % 3]))

    # Set labels and title
    ax.set_xlabel('rnn cells')
    ax.set_xlabel('f max')
    ax.set_title('')

    x_ticks_positions = np.arange(len(data_dict)) * 4 + 1.5
    ax.set_xticks(x_ticks_positions)
    ax.set_xticklabels(data_dict.keys())

    # Add legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    ax.legend(legend_handles, ['CN', 'MCI', 'AD'])

    plt.show()

#ordinal categorical crossentropy. Weighs error by how many classes output was off by. weights range from 1 to 2
# https://stats.stackexchange.com/questions/87826/machine-learning-with-ordered-labels
def occ_loss(y_true, y_pred, gamma=0):
    y_true_ord = tf.argmax(y_true, axis=1) #convert back to ordinal
    y_pred_ord = tf.argmax(y_pred, axis=1)

    if y_true_ord != 0:
        y_true_ord += gamma
    if y_pred_ord != 0:
        y_pred_ord += gamma

    weights = tf.cast(tf.abs(y_true_ord - y_pred_ord) / 2, dtype='float32') #2 = n_classes-1
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)

def find_prev_diffs(y_true, y_pred):

    if np.array_equal(np.unique(y_true), np.unique(y_pred)):
        y_true_vcs = np.array(pd.Series(y_true).value_counts().sort_index())
        y_pred_vcs = np.array(pd.Series(y_pred).value_counts().sort_index())

        return (y_true_vcs-y_pred_vcs) / y_true_vcs

    else:
        missing_classes = np.setdiff1d(np.unique(y_true), np.unique(y_pred))


def expectation(prob_vector):
    return round(sum([i*prob_vector[i] for i in range(len(prob_vector))]))