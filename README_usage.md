Voici le repository pour le server du projet DTY du Paris Digital Lab pour Sonergia.
Ce readme est destiné au personnes qui vont juste se servir des fonctionalités sans rentrer en détail dans l'implementation.
Plus de détail sont disponibles dans le README_dev ou dans la documentation technique.

# Installation

Avant d'installer l'environnement virtuel, il faut que vous avez python 3.8.5 d'installé et que vous utilisiez cette version de python pour lancer les commandes.

Vous pouvez vérifier la version de python avec :

```bash
python3 --version
```

Au début de l'installation vous pouvez créer un environnement virtuel avec la commande :

```bash
python3 -m venv .env
```

ajouter le nom de l'environnemment dans le .gitignore

Puis l'activer avec

windows:

```bash
.env\Scripts\activate.bat
```

macOS :

```bash
source .env/bin/activate

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

La documentation de l'api se trouve dans le fichier swagger.json sui se trouve dans le dossier src/app
