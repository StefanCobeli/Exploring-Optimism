import pandas as pd
from time import gmtime, strftime

def train_model(model, model_name, x_train, y_train\
                , x_dev=[None], y_dev=None\
                , epochs=1, batch_size=128\
                , models_path="./models/"\
                , trained_model_fn="trained_model_%s.h5"\
                , histories_path="./histories/"):

    # model.save(models_path + 'untrained_model.h5')

    model.compile(loss='binary_crossentropy', #"mean_squared_logarithmic_error", #'binary_crossentropy',
              optimizer='adam',#'adam',#'rmsprop',
              metrics=['accuracy'])

    print("Training model:")

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

    #Save trained model & Training history:
    val_acc    = "valAcc%.3f"%(training_history.history["val_acc"][-1])
    time_stamp = strftime("%H%M%S_%Y%m%d", gmtime())
    time_stamp = "_".join((model_name, val_acc, time_stamp))
    training_history_fn = histories_path + ("history_%s.csv" %time_stamp)
    pd.DataFrame(training_history.history)\
        .to_csv(training_history_fn)
    print("Training history saved in %s." \
            %(training_history_fn))

    trained_model_fn = models_path + trained_model_fn %time_stamp
    model.save(trained_model_fn)
    print("Trained model saved in %s." \
                %(trained_model_fn))
    return training_history, model
