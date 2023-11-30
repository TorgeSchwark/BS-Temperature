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

**Analyse von Architekturen (qualitativ, quantitativ):**
- MLP
- CNN
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