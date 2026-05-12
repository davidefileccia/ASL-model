# ASL Real-time Recognition

Riconoscimento in tempo reale dei segni del linguaggio dei segni americano (ASL) tramite webcam, utilizzando MediaPipe per il rilevamento dei landmark e un modello TensorFlow addestrato su [Google Isolated Sign Language Recognition (Kaggle)](https://www.kaggle.com/competitions/asl-signs).

---

## Requisiti di sistema

| Requisito | Dettagli |
|---|---|
| Sistema operativo | Windows 10 / 11 a 64 bit |
| Python | 3.11 (installato automaticamente da `setup.ps1`) |
| Webcam | Qualsiasi webcam USB o integrata |
| Connessione Internet | Necessaria solo durante il setup |

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
- Installazione di tutte le librerie necessarie

> Il setup richiede circa **5-10 minuti** a seconda della velocità della connessione (TensorFlow pesa ~350 MB).

---

## Avvio

Dopo aver completato il setup, avvia il progetto con:

```powershell
.\avvia.ps1
```

Si aprirà una finestra con il feed della webcam. Il modello inizierà a riconoscere i segni ASL una volta riempito il buffer (circa 13 secondi a 30 fps).

**Tasto `q`** — chiude l'applicazione.

---

## Librerie utilizzate

| Libreria | Versione | Scopo |
|---|---|---|
| `tensorflow` | 2.x | Caricamento ed esecuzione del modello di classificazione |
| `mediapipe` | 0.10.x | Rilevamento landmark di viso, mani e corpo in tempo reale |
| `opencv-python` | 4.x | Cattura video dalla webcam e rendering dell'interfaccia |
| `numpy` | 2.x | Elaborazione numerica degli array di landmark |

---

## Struttura del progetto

```
ASL model project/
├── asl_realtime_inference.py   # Script principale di inferenza
├── asl_model_complete.keras    # Modello Keras addestrato
├── asl_pure_tf_blackbox.zip    # Modello TensorFlow SavedModel (usato a runtime)
├── sign_to_prediction_index_map.json  # Mappatura indice → nome del segno (250 classi)
├── avvia.ps1                   # Script di avvio rapido
├── setup.ps1                   # Script di installazione automatica
└── Docs/                       # Documentazione aggiuntiva
```

---

## Come funziona

1. **MediaPipe Holistic** analizza ogni frame della webcam ed estrae 543 landmark (viso, mani, corpo)
2. I landmark vengono accumulati in un buffer di 384 frame (~13 secondi)
3. Ogni frame viene preprocessato con la stessa pipeline del modello Kaggle (normalizzazione, derivate temporali)
4. Il modello TensorFlow classifica la sequenza tra 250 segni ASL
5. Il segno riconosciuto viene mostrato sull'HUD in sovraimpressione

---

## Interfaccia

- **Barra superiore**: segno riconosciuto + percentuale di confidenza (verde = alta confidenza)
- **Barra inferiore**: avanzamento del buffer (diventa verde quando è pieno)
- Il riconoscimento parte automaticamente quando il buffer raggiunge i 384 frame

---

## Risoluzione problemi

| Problema | Soluzione |
|---|---|
| "python non trovato" dopo il setup | Riavvia PowerShell e riprova |
| Webcam non rilevata | Verifica che nessun altro programma stia usando la webcam |
| Errore TensorFlow all'avvio | Assicurati di aver eseguito `setup.ps1` e che `C:\asl_env` esista |
| Script bloccato da policy | Esegui `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
