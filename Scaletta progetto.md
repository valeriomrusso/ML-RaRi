# Scopo del progetto

Utilizzo di una libreria (tensorflow) e tre modelli (Ridge Regression "linear functions", NN con sigmoide (2 layer), RBF Radial basis function "non-linear functions").

Processo:
- Partizione dei dati
- Scelta dell'iperparametro
- Inizializzazione dei pesi
- Scadenza progetto: 29 gennaio ore 11:00
- Pretest: 12 febbraio



I risultati si hanno come risultato del TR, ma poi al prof dobbiamo dare i risultati sul TS.

Bisogna scegliere come partizionare di dati. le prove verranno fatto tramite 2 metodologie diverse: Hold-out, k-fold.
- Hold-out: TR, VL, TS con ripartizione base 60-20-20 (ma possiamo provare percentuali diverse es. 60-30-10, 50-30-20)
- K-fold: per quanto riguarda dataset piccoli conviene fare un leave-one-out (k = N, con N numero di dati)

Quali iperparametri vogliamo usare?
- Ridge: i parametri sono lambda (Tickonov) oppure usare LBE con phi (con random search dato che è più easy come ricerca degli iperparametri)
- NN:neuroni in un layer, learning rate, lamda (regolarizzazione), SGD, momentum
- RBF: i parametri sono lunghezza delle funzioni radiali (gamma), posizione dei centri (n_centers), numero delle unità nell'hidden layer, learning rate, lambda (si può fare una grid search per rendere computazionalmente più semplice la ricerca dell'iperparametro)

NOTA!! --> L'early stopping è consentito, nessuno ti dirà niente. Ma se viene colto in fallo un early stopping non necessario verrai buttato dal ponte di mezzo.

Inizializzazione dei pesi fatta con random search.
