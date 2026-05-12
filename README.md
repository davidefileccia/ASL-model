# ASL Real-time Recognition

Riconoscimento in tempo reale dei segni del linguaggio dei segni americano (ASL) tramite webcam, utilizzando MediaPipe per il rilevamento dei landmark e un modello TensorFlow addestrato sul dataset [Google Isolated Sign Language Recognition (Kaggle)](https://www.kaggle.com/competitions/asl-signs).

---

## Crediti

La rete neurale utilizzata in questo progetto è stata creata da **[hoyso48](https://www.kaggle.com/hoyso48)**, vincitore della competizione Kaggle *Google - Isolated Sign Language Recognition*. Il modello originale e il notebook di training sono disponibili sulla sua pagina Kaggle.

---

## Requisiti di sistema

| Requisito | Dettagli |
|---|---|
| Sistema operativo | Windows 10 / 11 a 64 bit |
| Python | 3.11 (installato automaticamente da `setup.ps1`) |
| Webcam | Qualsiasi webcam USB o integrata |
| Connessione Internet | Necessaria solo durante il setup |
| Spazio su disco | ~2 GB (Python + librerie + modello) |

---

## Installazione (prima esecuzione)

1. Apri **PowerShell** nella cartella del progetto
2. Se richiesto, abilita l'esecuzione degli script:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```
3. Esegui lo script di setup:
   ```powershell
   .\setup.ps1
   ```

Lo script esegue automaticamente:
- Download e installazione di **Python 3.11.9**
- Creazione del virtual environment in `C:\asl_env`
- Installazione di tutte le librerie con le versioni corrette

> Il setup richiede circa **5-10 minuti** (~500 MB di download).

> Il virtual environment viene creato in `C:\asl_env` invece che nella cartella del progetto per evitare problemi di encoding con percorsi che contengono caratteri speciali (es. `°`).

---

## Avvio

Dopo il setup, avvia il progetto con:

```powershell
.\avvia.ps1
```

Si aprirà una finestra ridimensionabile con il feed della webcam. Il programma parte in stato **IDLE** e attende che l'utente avvii la predizione.

### Stati del programma

| Stato | Descrizione |
|---|---|
| `IDLE` | Fotocamera attiva, nessun buffering. Mostrato solo al primo avvio |
| `BUFFERING` | Raccolta frame in corso. Il buffer si riempie fino a 384 frame (~13s a 30fps) |
| `PREDIZIONE` | Il risultato viene mostrato sull'HUD e stampato in console |

### Tasti

| Tasto | Quando | Funzione |
|---|---|---|
| `N` | In idle | Avvia il buffering |
| `SPAZIO` | Durante buffering | Predizione immediata: ripete i frame catturati fino a 384 e lancia l'inferenza. Utile per segni statici o brevi — bastano 1-2 secondi |
| `L` | Sempre | Attiva/disattiva la visualizzazione dei landmark MediaPipe |
| `Q` | Sempre | Chiude l'applicazione |

> **Consiglio per segni dinamici** (es. "yes", "no"): prima di premere `SPAZIO` cattura almeno 2-3 cicli completi del movimento, altrimenti la ripetizione dei frame potrebbe non contenere abbastanza informazione sul pattern.

### Output in console

Lo script stampa lo stato del programma direttamente in console:
```
[status] IDLE — fotocamera avviata. Premi N per avviare.
[status] BUFFERING — nuova raccolta avviata.
[predizione] nose  (87%)
[status] IDLE — predizione inviata. Premi N per una nuova predizione.
[predizione] yes  (72%)
```

---

## Librerie utilizzate

Le versioni indicate sono quelle testate e verificate. **Non aggiornare `tensorflow` e `mediapipe` arbitrariamente**: versioni più recenti di mediapipe (≥ 0.10.20 circa) hanno rimosso l'API `mp.solutions` usata dallo script, e versioni più recenti di TensorFlow causano conflitti su `protobuf` e `jax`.

| Libreria | Versione | Scopo |
|---|---|---|
| `tensorflow` | 2.17.0 | Caricamento ed esecuzione del modello di classificazione |
| `mediapipe` | 0.10.14 | Rilevamento landmark di viso, mani e corpo in tempo reale |
| `opencv-python` | ultima stabile | Cattura video dalla webcam e rendering dell'interfaccia |
| `numpy` | compatibile con TF 2.17 | Elaborazione numerica degli array di landmark |

---

## Struttura del progetto

### File necessari

| File | Scopo |
|---|---|
| `asl_realtime_inference.py` | Script principale di inferenza |
| `asl_pure_tf_blackbox.zip` | Modello TensorFlow SavedModel usato a runtime |
| `sign_to_prediction_index_map.json` | Mappatura indici → nomi dei 250 segni ASL |
| `avvia.ps1` | Script di avvio rapido |
| `setup.ps1` | Script di installazione automatica dell'ambiente |
| `.gitignore` | Esclude dal repository i file generati automaticamente |
| `README.md` | Documentazione del progetto |

### File superflui al funzionamento

| File/Cartella | Motivo |
|---|---|
| `asl_model_complete.keras` | Lo script usa solo `asl_pure_tf_blackbox.zip` — questo file non viene mai caricato a runtime |
| `_savedmodel_extracted/` | Generata automaticamente dallo script al primo avvio estraendo lo ZIP — può essere cancellata, verrà ricreata |
| `Docs/MITA_Sign_Recognition.ipynb` | Notebook di sviluppo, non usato a runtime |
| `.claude/` | File interni di Claude Code, non pertinenti al progetto |

---

## Come funziona

1. **MediaPipe Holistic** analizza ogni frame della webcam ed estrae 543 landmark (viso, mani, corpo)
2. I landmark vengono accumulati in un buffer di 384 frame (~13 secondi a 30 fps)
3. Ogni frame viene preprocessato con la stessa pipeline del modello originale (normalizzazione Z-score, derivate temporali dx e dx²)
4. Il modello TensorFlow classifica la sequenza tra 250 segni ASL
5. Il segno riconosciuto viene mostrato sull'HUD in sovraimpressione

---

## Interfaccia

- **Barra superiore**: segno riconosciuto + percentuale di confidenza
  - Verde: confidenza ≥ 40%
  - Azzurro: confidenza sotto soglia (`???`)
- **Barra inferiore**: avanzamento del buffer (blu durante il riempimento, verde quando pieno) con promemoria dei tasti disponibili
- Il riconoscimento parte automaticamente quando il buffer raggiunge i 384 frame, oppure manualmente con `SPAZIO`

---

## Risoluzione problemi

| Problema | Soluzione |
|---|---|
| Script bloccato da policy | Esegui `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| "python non trovato" dopo il setup | Riavvia PowerShell e riprova |
| Webcam non rilevata | Verifica che nessun altro programma stia usando la webcam |
| Errore `mp.solutions` non trovato | Non aggiornare mediapipe oltre la 0.10.14; riesegui `setup.ps1` |
| Errore `protobuf` o `jax` | Versioni incompatibili installate; cancella `C:\asl_env` e riesegui `setup.ps1` |
| Modello non trovato | Assicurati che `asl_pure_tf_blackbox.zip` sia nella cartella del progetto |
