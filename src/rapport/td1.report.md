# Partie 1: Text classification: prédire si la vidéo est une chronique comique

Collas Pierre / Singarayar Jeffrey

## Tasks

### Adapter "src/" pour que la pipeline "python src/main.py evaluate" marche sur la task 

Dans le code du projet, la pipeline evaluate a été modifié de manière à faire le dataset , faire les features , créer le model puis l'évaluer.

```python
@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):

    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task)

    # Object with .fit, .predict methods
    model = make_model()

    # Run k-fold cross validation. Print results
    return model.evaluate(X, y)
```

### Regarder les features disponibles dans sklearn.feature_extraction.text. Lesquelles semblent adaptées ?

CountVectorizer: Convertit une collection de documents texte en une matrice de comptages de tokens.

TfidfVectorizer: Convertit une collection de documents texte en une matrice de fonctionnalités TF-IDF.

TfidfTransformer: Transforme une matrice de comptage en une représentation TF ou TF-IDF.

HashingVectorizer: Convertit une collection de documents texte en une matrice de token occurrences.

FeatureHasher: Effectue une hachage de caractéristiques sur un ensemble de documents.

Parmi les features disponibles dans sklearn.feature_extraction.text présentés ci-dessus, TfidfVectorizer est souvent un choix standard pour le traitement de texte car il prend en compte non seulement la fréquence des mots dans un document particulier (TF) mais aussi la fréquence inverse du document (IDF) dans l'ensemble du corpus, ce qui pourrait aider à minimiser l'impact des mots communs et à mettre en évidence les termes plus distinctifs.

CountVectorizer est également un bon choix assez classique pour convertir le texte en vecteurs où l'importance d'un mot est proportionnelle à sa fréquence dans le document.

HashingVectorizer est similaire à CountVectorizer mais est utilisé quand nous avons un grand vocabulaire et que nous voulons réduire la dimensionnalité. Dans notre cas de figure, les noms de vidéos n'étant pas de très grande taille, il ne nous a pas semblé très pertinent.

TfidfTransformer est utilisé pour transformer les vecteurs de fréquence en vecteurs TF-IDF après avoir utilisé CountVectorizer.

TfidfVectorizer est utilisé pour convertir le texte en vecteurs TF-IDF en une seule étape ce qui en fait une alternative plus intéressante que le TfidfTransformer.

### Documentation 

Dans la modification de la pipelie, nous avons fait en sorte d'ajouter un fichier "config.json" qui permettra de choisir avant de lancer une commande train, test, evaluate de choisir :

- Le type de model qui sera utilisé (obligatoire):
    - LogisticRegression
    - RandomForestClassifier

- Le type de vectorisation (obligatoire):
    - CountVectorizer
    - TfidfVectorizer

- Les opérations de preprocessing (optionnel):
    - lowercase : retire toutes les majuscules
    - punctuation : retire les ponctuations
    - accent : retire les accents
    - stemming : ajoute du steeming

- Les extractions de features (optionnel): 
    - video_name_length : la longueur de caractère du nom de la vidéo
    - video_name_word_count : le nombre de mots de la vidéo
    - num_uppercase : le nombre de majuscules dans le nom de la vidéo
    - has_number : la présence de nombre dans le nom de la vidéo

### Les a-priori que vous aviez sur les features & modèles utiles ou non

Les a-priori que nous avions sur les différentes méthodes de preprocessing et de features était dans un premier temps qu'elles allaient avoir un réel impact sur notre metric de performance.

Nous pensions ainsi qu'ajouter ou retirer une feature ou une opération de preprocessing allait significativement modifier la metric.

La modification significative de performance face à l'ajout ou la suppression de l'une d'elles, aurait pu attester de son importance pour le modèle.

Egalement, nous pensions qu'augmenter le ngram_range de manière capter des phrases ou expressions composées permettrai d'augmenter les performances notemment dans le cas d'expressions humouristiques.

### Quels ont été les apports individuels de chacune de ces variation ?

Suite à une multitude de tests avec une multitude de combinaisons de features ou opérations de preprocess et même de modèles différents, nous avons pu observer d'après nos résultats qu'aucune d'elles n'ont de réels apports significatifs à la performance du modèle. Entre les différentes évaluations nous pouvons observer des légères modifications de l'accuracy néanmoins, celles-ci semblent être plus dû à la part de variabilité engendrée par la crossvalidation que par les modifications elles-mêmes.

Pour illustrer cela, nous allons d'abord évaluer les performances d'un model LogisticRegression sans features apportés en plus ni preprocessing. 

```json
{
    "vectorizer": {
        "method": "TfidfVectorizer"
    },
    "features": [
            
        ],
    "preprocess_operations": [
        
    ],
    "model": {
        "type": "LogisticRegression"
    }
}
```
RESULT : Got accuracy 87.39246231155778%

A présent, en ajoutant les features :

```json
{
    "vectorizer": {
        "method": "TfidfVectorizer"
    },
    "features": [
            "video_name_length",
            "video_name_word_count",
            "num_uppercase",
            "has_number"
        ],
    "preprocess_operations": [
        
    ],
    "model": {
        "type": "LogisticRegression"
    }
}
```

RESULT : Got accuracy 88.69497487437187%

Nous pouvons observer une légère augmentation de l'accuracy avec l'ajout de ces 4 features
A présent, si nous ajoutons le preprocessing :

```json
{
    "vectorizer": {
        "method": "TfidfVectorizer"
    },
    "features": [
            "video_name_length",
            "video_name_word_count",
            "num_uppercase",
            "has_number"
        ],
    "preprocess_operations": [
        "lowercase",
        "punctuation",
        "accent",
        "stemming"
    ],
    "model": {
        "type": "LogisticRegression"
    }
}
```

RESULT : Got accuracy 88.39195979899498%

Les preprocessing dont le steeming ne semble donc pas avoir eu d'impact sur les performances.

A présent, si nous passons à un CountVectorizer

```json
{
    "vectorizer": {
        "method": "CountVectorizer"
    },
    "features": [
            "video_name_length",
            "video_name_word_count",
            "num_uppercase",
            "has_number"
        ],
    "preprocess_operations": [
        "lowercase",
        "punctuation",
        "accent",
        "stemming"
    ],
    "model": {
        "type": "LogisticRegression"
    }
}
```
RESULT : Got accuracy 90.69346733668343%

Nous avons pu observer une légère amélioration des performances lors de l'utilisation d'un CountVectorizer avec un modèle LogisticRegression comparativement à un TfidfVectorizer


Enfin, nous allons maintenant effectuer la même batterie d'experimentations pour un modèle de RandomForestClassifier:

Commençons sans features ni preprocess :

```json
{
    "vectorizer": {
        "method": "TfidfVectorizer"
    },
    "features": [
            
        ],
    "preprocess_operations": [
    
    ],
    "model": {
        "type": "RandomForestClassifier"
    }
}
```

RESULT : Got accuracy 89.89145728643216%

Nous pouvons notemment observer quand dans la configuration similaire avec un modèle de regression logistique, les performances sont cette fois-ci meilleures.


En ajoutant les features à présent :

```json
{
    "vectorizer": {
        "method": "TfidfVectorizer"
    },
    "features": [
            "video_name_length",
            "video_name_word_count",
            "num_uppercase",
            "has_number"
        ],
    "preprocess_operations": [

    ],
    "model": {
        "type": "RandomForestClassifier"
    }
}
```

RESULT : Got accuracy 90.49346733668342%

Les performances sembles cette fois-ci légèrement supérieures


Si nous ajoutons à présent les preprocess :

```json
{
    "vectorizer": {
        "method": "TfidfVectorizer"
    },
    "features": [
            "video_name_length",
            "video_name_word_count",
            "num_uppercase",
            "has_number"
        ],
    "preprocess_operations": [
        "lowercase",
        "punctuation",
        "accent",
        "stemming"
    ],
    "model": {
        "type": "RandomForestClassifier"
    }
}
```

RESULT : Got accuracy 90.99145728643215%

Les perfomances sont semblables à la configuration précedente, ne montrant vraisemblabement pas d'apport particulier.

Enfin, si nous utilisons maintenant un CountVectorizer :

```json
{
    "vectorizer": {
        "method": "CountVectorizer"
    },
    "features": [
            "video_name_length",
            "video_name_word_count",
            "num_uppercase",
            "has_number"
        ],
    "preprocess_operations": [
        "lowercase",
        "punctuation",
        "accent",
        "stemming"
    ],
    "model": {
        "type": "RandomForestClassifier"
    }
}
```

RESULT : Got accuracy 89.8894472361809%

Les performances sembles similaires voir inférieurs compartivement à l'utilisation d'un TfidfVectorizer.

En somme, parmi les différents tests effectués, la configuration démontrant les performances les plus élevées est la suivante (bien que les apports soient relativement minimes) : 

```json
{
    "vectorizer": {
        "method": "TfidfVectorizer"
    },
    "features": [
            "video_name_length",
            "video_name_word_count",
            "num_uppercase",
            "has_number"
        ],
    "preprocess_operations": [
        "lowercase",
        "punctuation",
        "accent",
        "stemming"
    ],
    "model": {
        "type": "RandomForestClassifier"
    }
}
```

Gardons à l'esprit que certaines différences ont de grandes chances d'être dû à la crossvalidation et que l'apport réel est extrèmement faible. Pour autant, bien que l'apport soit faible, la quantité de ressource utilisé pour effectuer ces calculs n'est elle absolument pas négligeable.

De ce fait, dans un context où nous aurions des ressources de calculs extrèmement limité, il serait donc necessaire de considérer le fait que le gain minime de performance ne vaut peux être pas la charge suplémentaire de calcule à effectuers.

Notre hypothèse principale pour expliquer ces résultats est que les nouvelles caractéristiques introduites peuvent ne pas contenir d'informations utiles pour prédire notre variable.
Egalement, certaines caractéristiques peuvent davantage ajouter du bruit plutôt que des informations utiles.

Notons également que nous avons essayé de jouer avec le paramètre ngram_range du TfidfVectorizer pour pouvoir essayer dans certains cas d'aider le modèle à capturer des informations contextuelles plus larges, comme les phrases ou expressions composées de plusieurs mots. 

Seulement, l'augmentation de ngram_range=(1, 1) à ngram_range=(1, 2) , ngram_range=(1, 3) , ngram_range=(1, 4) a relativement dégradé les performances du modèle en le faisant retomber sous la barre des 0,87 d'accuracy.

D'autre part, si le modèle est trop simple, il pourrait ne pas être capable de capturer les informations utiles fournies par les nouvelles caractéristiques. Il est ainsi possible que cela en soit la cause.

Enfin, dans une logique de perspective d'amélioration, il aurait pu être interessant d'ajouter un type d'extraction de feature se concentrant sur la présence de mots clés faisant parti d'un champ lexical lié à l'humour.

### Adapter "src/" pour que les pipelines "train" et "predict" marchent

Les pipelines train et predict ont été modifiées de manière à ce que train puisse entrainer un model et finir par le dump et preidct puisse récupérer un modèle dump pour effectuer des prédictions, le tester et renvoyer un fichier .csv avec les rédictions.




