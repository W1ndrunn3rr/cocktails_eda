
# EDA i Klastrowanie zbioru danych cocktail_dataset.json

Raport EDA oraz ewaluacja wyników klastrowania znajduje się w pliku o ścieżce 
```bash
../notebooks/raport.ipynb
```
---



W celu instalacji bibliotek, nalezy stworzyć na początek środowisko wirtaulne Pythona z użyciem venva
```bash
python -m venv eda_venv
```
Następnie aktywować venva: 

1.Windows
```bash
../eda_venv/Scripts/activate
```

2.Linux / MacOS
```bash
../eda_venv/bin/activate
```

Teraz można zainstalować paczki za pomocą menegera pip :
1.Przejdź do folderu z projektem i wywołaj :
```bash
pip install -r requirements.txt
```

Projekt jest gotowy do wywołania

Należy przejść do folderu scirpts i wywołać :
``` bash
python main.py
```

Skutkiem działania programu będzie ewaluacja wyniku modelu oraz zapis danych w tabeli csv, który można użyć w dalszej obróbce lub wizualizacji, ewentualnie w uczeniu nadzorowanym.

