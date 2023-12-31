# Phase 2

## Anmerkung Datenverarbeitung (nicht explizit gefordert):
- Temperatur Daten werden auf einen Float mit 2 Nachkommastellen gerundet
- diese Daten werden wiederum durch das data_processing.py pro Land, Staat, Stadt, Hauptstadt in separate txt Dateien geladen, sortiert nach Datum
  - Idee zur Verbesserung: Daten einlesen und in einem Array speichern, somit müssen bei jedem aufruf des Data generators keine Dateien eingelesen werden sondern nur auf das Array im RAM zugegriffen werden

## Fragestellungen:

**Implementierung Data-Loader Pipeline:**
- aktuelles gegebenes Vorgehen
  - ```data_generator``` $\rightarrow$ ```select_data``` $\rightarrow$ ```parse_file```, ```find_starting_idices```, ```autmengation-part```
- siehe code TODO: (optimieren)
- Erster Verbesserungsversuch durch einmaliges Einlesen aller Daten hat keine erkennbare Verbesserung gebracht (TODO: oder?)
- Dataloader so modifizieren, dass ALLE daten genommen werden (pro Epoche?)

**Visualisierungen:**

- Histogramme:

| longitude | latitude |
| -- | -- |
| ![longitude histogramm](.\\histograms\\Longitude.png) | ![latitude histogramm](.\\histograms\\Latitude.png) |
| ![Continents](.\\histograms\\Continents.png) | ![Gaps in Data](.\\histograms\\Data_gaps.png) |
| ![Temperatur](.\\histograms\\Temperatur.png) | ![Uncertainty over years](.\\histograms\\Uncertainty_per_year.png) |
| ![Uncertainty](.\\histograms\\Uncertainty.png) | ![Years](.\\histograms\\Years.png) |

- Map Plot of data samples:  
![map plot](.\\histograms\\map_plot_data_points.png)

<!-- Transformers: https://bbycroft.net/llm -->

Datenanalyse:
- Städte haben teilweise exakt die gleichen Werte (in mehreren locations??)
- siehe ```analyse_data.py``` function (```analyse_same_val_cities()```)

**Analyse von Architekturen (qualitativ, quantitativ):**
- MLP
  - Activation functions
    - Selu
    - Relu
      - größtenteils benutzt, funktioniert gut (besser als andere?)
- CNN (1D)
  - mit global average pooling vor den dense layern
    - mit Learning rate bei 0.1 sah man solche Werte wie:
      - Epoch 24: val_mae did not improve from 25.38875
        30/30 [==============================] - 16s 555ms/step - loss: 2934.5496 - mse: 2934.5491 - mae: 42.1311 - val_loss: 1541.8851 - val_mse: 1541.8850 - val_mae: 32.9383
    - mit anderen learning rates (0.01, 0.001) liegen die Werte immer meistens bei: 
      - Epoch 6: val_mae did not improve from 4.40800
        100/100 [==============================] - 45s 451ms/step - loss: 47.1001 - mse: 47.1001 - mae: 5.1008 - val_loss: 38.9700 - val_mse: 38.9700 - val_mae: 4.5528
  - mit flatten vor den dense layers
    - vergleichsweise richtig gute Ergebnisse
    - TODO: siehe grafiken
    - allgemein: 
      - ab einem bestimmten Punkt performanen ALLE Modelle plötzlich deutlich besser
      - GRÖ?größere Filter könnten sinnvoll sein, da dann z.B. ein ganzes jahr betrachtet werden kann
- RNN
  - ! exploding gradients (kann durch gradient clipping verhindert werden)
- LSTM
  - nicht lineares Verhalten beim mea
- Transformer
  - Errors (out of memory)
    - bei einer Konfiguration mit (und den Standard SEQ_LEN etc.):
      - ```batch_size = 2```
      - ```learning_rate = 0.0001```
      - ```dropout = 0```
      - ```num_transformer_blocks = 8```
      - ```mlp_units=[128]```
      - mit ```batch_size = 1``` $\Rightarrow$ trotzdem ein MAE von ~2.35 bei 50 Epochen, 50 Steps
- ...
- Allgemein:
  Sehr häufig kam es bei den Modellen (abgesehen von MLP) immer wieder zu einer Grenze vom MAE bei 5, unsere Modelle kamen also nur in kurzen Ausnahme zu MAE Werten von gering unter 5 (dabei liegen mse bei ~30-40)
  TODO: bei kleinerer learning rate auch steps erhöhen (um gleiche Anzahl an samples zu haben)
  TODO: lineare activation function ausprobieren
- Bewertung Architekturen:
  - Qualitativ:
    - MLPs (TODO: oder LSTM?) performen nach detailierter Suche von allen am besten
  - Quantitativ:
    - Transformer (TODO: oder MLP?) performen im Schnitt am besten ohne viel am Code vornehmen zu müssen


positional encoding (sin wird iwi in Daten eingebracht?):
- interessant für:
  - CONV
  - Transformer

**Analyse mit und ohne Normalisierung der Daten:**
- ... TODO:

**Performance bei variablen Input-Sequenzen (8, 16, 32, 64):**
- 8:
- 16:
- 32:
- 64:

## Weitere Ideen:
- (later optional): try a binary model (so only binary weights, inputs, outputs, etc.)  
    -> e.g. for tf dataset can be casted (normalized) to specific values (float32, int, bool?)
- instegesammt nutzbare Datenzeiträume 4.071.935 !!

## Some Data Collection
- Eigene predictions machen um eine Menschliche Vorhersage mit der eines NN zu vergleichen

## Data Augmentation
- skalieren und transformieren
  - um Offset verschieben
  - ...

## Error Metriken / Vergleichparameter (Güte von nNs vergleichen)
- mse: nicht so gut geeignet, da fehler stärker bestraft werden bei sehr variablen Temperaturen  
  $L_{MSE}(w, y, \hat y)= \frac 1 {2p} \sum^p_{\mu = 1} L^{\mu}_{MSE}(w,y,\hat y) = \frac 1 {2p} \sum^p_{\mu = 1} \hat y(w, x^{(\mu)}-y^{\mu})^2$
- mae: eher geeignet  
  $MAE=\frac {\sum^n_{i=1} |y_i - x_i|} n = \frac {\sum^n_{i=1}|e_i|} n$

## NAS (neoronal architecture search)
- automatisierte Model Suchen werden sich höchstwahrscheinlich nicht lohnen, da der Searchspace einfach zu groß ist (z.B. im Gegenteil zu Edge Devices)
- kann ggf. trotzdem ausprobiert werden

# Phase 3
> Dokumentation von Herangehensweise, Aufgaben und Expirimenten
> Wissenschaftliches Vorgehen und Schreiben

## Conv net:
- bei konstanter epoch zahl
- 3-4 verschiedne architekturen
- batch size [50,100] ? so mehr so besser
- 3 learning rates [0.01,0.005,0.001] ?? 
- dann potten 
 
TODO:
- Visualisierung von predictions (Temperatur der Monate)
- Zukunftspredictions



Paper: https://www.overleaf.com/project/656739453dc04ca9c1e95e41