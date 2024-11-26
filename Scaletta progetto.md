# Scopo del progetto

Utilizzo di 2 librerie (tensorflow e pytorch) e tre modelli (Ridge Regression "linear functions", NN con sigmoide (2 layer), RBF Radial basis function "non-linear functions").

Processo:
- Partizione dei dati
- Scelta dell'iperparametro
- Inizializzazione dei pesi
- Creazioni reti (divisione del lavoro -> prima prova su tensorflow sui 3 modelli "Alex-NN, Michele-Ridge, Valerio-RBF" e poi si passa a pytorch scambiandosi i modelli)
- Scadenze progetti: 7 gennaio, 29 gennaio
- Esame: 22 gennaio, 12 febbraio



I risultati si hanno come risultato del TR, ma poi al prof dobbiamo dare i risultati sul TS.

Bisogna scegliere come partizionare di dati. le prove verranno fatto tramite 2 metodologie diverse: Hold-out, k-fold.
- Hold-out: TR, VL, TS con ripartizione base 60-20-20 (ma possiamo provare percentuali diverse es. 60-30-10, 50-30-20)
- K-fold: per quanto riguarda dataset piccoli conviene fare un leave-one-out (k = N, con N numero di dati)

Quali iperparametri vogliamo usare?
- Ridge: i parametri sono lamda (Tickonov) oppure usare LBE con phi (con random search dato che è più easy come ricerca degli iperparametri)
- NN:neuroni in un layer, learning rate, lamda (regolarizzazione), SGD, momentum
- RBF: i parametri sono lunghezza delle funzioni radiali (sigma), posizione dei centri (C), numero delle unità nell'hidden layer, learning rate, lamda (si può fare una grid search per rendere computazionalmente più semplice la ricerca dell'iperparametro)

NOTA!! --> L'early stopping è consentito, nessuno ti dirà niente. Ma se viene colto in fallo un early stopping non necessario verrai buttato dal ponte di mezzo.

Inizializzazione dei pesi fatta con random search.
