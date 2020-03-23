import pandas as pd
from time import gmtime, strftime

def train_model(model, model_name, x_train, y_train\
                , x_dev=[None], y_dev=None\
                , epochs=1, batch_size=128\
                , models_path="./models/"\
                , trained_model_fn="trained_model_%s.h5"\
                , histories_path="./histories/"):

    # model.save(models_path + 'untrained_model.h5')

    #'adam', "adadelta", 'adagrad', "adamax", 'rmsprop', "nadam"
    model.compile(loss='binary_crossentropy', #"mean_squared_logarithmic_error", #'binary_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

    print("Training %s..." %model_name)

    if not(None in x_dev):
        training_history = model.fit(#[x_train_seq, x_train_hot], y_train,
                            x_train,\
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_dev, y_dev))
    else:
        training_history = model.fit(#[x_train_seq, x_train_hot], y_train,
                            x_train,\
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs)

    print("%s trained!" %model_name)

    return training_history, model
