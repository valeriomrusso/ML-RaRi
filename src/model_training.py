import keras_tuner as kt
import keras
import tempfile
from model_builder import build_model_nn_ranged_tuner, build_model_ridge_ranged_tuner

class Random(kt.tuners.RandomSearch):
  def run_trial(self, trial, *args, **kwargs):
    kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 8, 32, step = 2)
    #kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
    return super(Random, self).run_trial(trial, *args, **kwargs)
  
'''class Grid(kt.tuners.GridSearch):
  def run_trial(self, trial, *args, **kwargs):
    kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 8, 32, step = 8)
    #kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
    return super(Grid, self).run_trial(trial, *args, **kwargs)'''

def train_model_ranged(fold_no, build_model, x_train, y_train, x_val, y_val, task):
    """Training con gestione errori"""
    try:
        if build_model == build_model_nn_ranged_tuner:
            build_fn = build_model_nn_ranged_tuner(task)
        elif build_model == build_model_ridge_ranged_tuner:
            build_fn = build_model_ridge_ranged_tuner(task)
        if task == 'CUP':
            objective = 'val_loss'
        elif task == 'MONK':
            objective = 'val_accuracy'
        temp_dir = tempfile.mkdtemp()
        
        tuner = Random(
        build_fn,
        objective=objective,
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
        model = build_fn(best_hps)
        
        return model, tuner
        
    except Exception as e:
        print(f"Errore durante il training: {str(e)}")
        raise

def train_model_fixed(model, batch_size, X_train, X_val, Y_train, Y_val):
    final_history = model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=2000,  # Puoi regolare il numero di epoche
        validation_data=(X_val, Y_val),  # Usa il test set per la validazione finale
        callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=5)]
    )

    return final_history