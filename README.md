# Projekt 2: Hyperparameter Optimization

## Einrichtung
Zunächst muss eine venv mit Python Version 3.11 erstellt werden.

```
py -3.11 -m venv hyper_env
hyper_env\Scripts\activate # für windows
source hyper_env/bin/activate # für linux oder mac
pip install -r requirements.txt
```

Hinweis:
Um die bestmögliche Performance zu erreichen, wird cuda auf einer GPU genutzt. Dafür ist eine manuelle Auswahl der für die GPU passenden torchh Version notwendig. Die requirements.txt installiert nur die CPU Version von torch!

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Ausführung

Zum Laufen lassen und Vergleichen der Optimizer wird das Tournament Notebook (`notebooks\tournament.ipynb`) verwendet.

## Hinzufügen von Modellen und Optimizern
Modelle und Optimizer müssen in die jeweiligen Ordner `models` bzw. `optimizers` gepackt werden. Dabei können jeweils auch Unterordner erstellt werden.
Optimizer und Modelle müssen auch jeweils das Wort "Optimizer" bzw. "Model" im Namen sowohl vom File, als auch von der Klasse, haben und die jeweiligen Base-Klassen implementieren.
Wenn all dies erfüllt ist, werden die Optimizer/ Modelle automatisch als solche vom Tournament Notebook erkannt.
Beispiele sind in den Ordnern hinterlegt.