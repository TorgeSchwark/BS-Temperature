# Phase 2

## Anmerkung Datenverarbeitung (nicht explizit gefordert):
- Temperatur Daten werden auf einen Float mit 2 Nachkommastellen gerundet
- diese Daten werden wiederum durch das data_processing.py pro Land, Staat, Stadt, Hauptstadt in separate txt Dateien geladen, sortiert nach Datum
  - Idee zur Verbesserung: Daten einlesen und in einem Array speichern, somit müssen bei jedem aufruf des Data generators keine Dateien eingelesen werden sondern nur auf das Array im RAM zugegriffen werden

## Fragestellungen:

**Implementierung Data-Loader Pipeline:**
- siehe code TODO: (optimieren)
- Erster Verbesserungsversuch durch Einmaliges Einlesen aller Daten hat keine erkennbare Verbesserung gebracht

**Visualisierungen:**

- Histogramme:

| longitude | latitude |
| -- | -- |
| ![longitude histogramm](.\\histograms\\Longitude.png) | ![latitude histogramm](.\\histograms\\Latitude.png) |
| ![Continents](.\\histograms\\Continents.png) | ![Gaps in Data](.\\histograms\\Data_gaps.png) |
| ![Temperatur](.\\histograms\\Temperatur.png) | ![Uncertainty over years](.\\histograms\\Uncertainty_per_year.png) |
| ![Uncertainty](.\\histograms\\Uncertainty.png) | ![Years](.\\histograms\\Years.png) |
... TODO:

Datenanalyse:
- Städte haben teilweise exakt die gleichen Werte (in mehreren locations??)
- siehe ```analyse_data.py``` function (```analyse_same_val_cities()```)

**Analyse von Architekturen (qualitativ, quantitativ):**
- MLP
- CNN (1D)
  - mit Learning rate bei 0.1 sah man solche Werte wie:
    - Epoch 24: val_mae did not improve from 25.38875
      30/30 [==============================] - 16s 555ms/step - loss: 2934.5496 - mse: 2934.5491 - mae: 42.1311 - val_loss: 1541.8851 - val_mse: 1541.8850 - val_mae: 32.9383
- RNN
- ...

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

## Some Data Collection
- Eigene predictions machen um eine Menschliche Vorhersage mit der eines NN zu vergleichen


# Phase 3
> Dokumentation von Herangehensweise, Aufgaben und Expirimenten
> Wissenschaftliches Vorgehen und Schreiben





Paper: https://www.overleaf.com/project/656739453dc04ca9c1e95e41