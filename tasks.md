## Netzwerkarchitekturen

Im Rahmen des Seminars sollen verschiedene Netzwerkarchitekturen untersucht werden. Speziell sollen die folgenden Architekturen zum Einsatz kommen:  
- MLP
- 1D Conv Net
- LSTM
- Transformer

In Phase 2 geht es um:  
- Bekannt machen mit einem oder mehrerer der 6 Datensätze. Analyse der Daten und Organisieren der Inputs/Outputs.
- Trainieren mit allen NN-Architekturen.
- Implementierung verschiedener Visualisierungen für die Analyse der Daten/Predictions.
Training und Analyse auf den Daten.
Folgende Fragestellungen sind zu bearbeiten:
Implementiere eine Data-Loader Pipeline für den von dir ausgewählten Datensatz. Die Daten-Pipeline soll Sequenzen von Datenpunkten als Mini-Batches (wie im Code-Beispiel gezeigt) bereitstellen.
- Implementiere Code für das Visualisieren der Daten (e.g., Histogramme, t-SNE Plots). Welche Daten der Datensätze können wie untersucht und visualisiert werden?
Welche Architektur performt am besten (qualitativ, quantitativ)?
Normalisiere die Daten vor dem Trainieren deines Netzwerks und vergleiche die Vorhersagegenauigkeit, wenn das Netzwerk mit oder ohne normalisierte Daten trainiert wird.
- Wie verhält sich die Performance einer beliebigen Architektur, wenn die Länge der Input-Sequenzen von 8, 16, 32, 64, verändert wird?