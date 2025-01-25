import keras_tuner as kt
import keras
import tempfile

def train_model(fold_no, build_model, x_train, y_train, x_val, y_val):
    """Training con gestione errori"""
    try:
        temp_dir = tempfile.mkdtemp()
        tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=50,  # Più prove per una maggiore diversità
        overwrite=True,
        directory=temp_dir,
        project_name=f'fold_{fold_no}'
        )
        
        tuner.search(
            x_train, y_train,
            epochs=50,
            validation_data=(x_val, y_val),
            callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=5)]
        )
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = build_model(best_hps)
        
        return model, tuner
        
    except Exception as e:
        print(f"Errore durante il training: {str(e)}")
        raise