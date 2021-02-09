Voici le repository pour le server du projet DTY du Paris Digital Lab pour Sonergia.
Ce readme est destiné au personnes qui vont juste se servir des fonctionalités sans rentrer en détail dans l'implementation.
Plus de détail sont disponibles dans le README_dev ou dans la documentation technique.

# Installation

Au début de l'installation vous pouvez créer un environnement virtuel avec la commande :

```bash
python3 -m venv nom
```

Puis l'activer avec

windows:

```bash
nom\Scripts\activate.bat
```

macOS :

```bash
source nom/bin/activate

```

Pour installer toutes les dependances nécéssaires à ce projet lancé la commande suivante:

```bash
pip install -r requirements.txt
```

Pour ensuite lancer le server utilisez les commandes suivantes :

```bash
cd src/app

uvicorn app:app
```

# Documentation des differentes routes de l'api
