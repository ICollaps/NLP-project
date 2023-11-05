# Rapport TD 2 - 3

## Modèle : is_name

Nous avons commencé par modifier certaines parties du code de manière à pouvoir convenablement intégrer ce second modèle. Ainsi, après avoir fait ces changements nous avons pu effectuer différents tests.

Pour le second model en intégrant les features nous obtenons une bonne accuracy sur le model cependant, en observant le rapport de classification, nous pouvons constater que les performances concernant la classe "1" sont mauvaises. Ces résultats sont typiques d'un cas où nous travaillions sur un dataset déséquilibré. Et en effet, nous avons beaucoup plus de mots ne correspondant pas à des noms que l'inverse dans le dataset.

```
Cross-validated accuracy: 95.96%
Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.99      0.98     11815
           1       0.42      0.09      0.15       481

    accuracy                           0.96     12296
   macro avg       0.69      0.54      0.56     12296
weighted avg       0.94      0.96      0.95     12296
```

Sans l'utilisation des features "is_final_word", "is_capitalized", nous pouvons d'ailleurs voir que l'accuracy du model reste sensiblement la même mais que les performances relatives à la classe 1 sont encore inférieures.

```
Cross-validated accuracy: 95.85%
Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.99      0.98     11815
           1       0.38      0.09      0.15       481

    accuracy                           0.96     12296
   macro avg       0.67      0.54      0.56     12296
weighted avg       0.94      0.96      0.95     12296```

Ainsi, en intégrant une rééquilibration du dataset visant à sous-échantillionner la classe 0 nous obtenons les résultats suivants (sans features) :

```
Classification Report:
              precision    recall  f1-score   support

           0       0.55      0.99      0.70       481
           1       0.95      0.18      0.30       481

    accuracy                           0.59       962
   macro avg       0.75      0.59      0.50       962
weighted avg       0.75      0.59      0.50       962
```

Les performances globales du modèle ont chuté. cependant, en ajoutant les features et en conservant le rééquilibrage du dataset, nous obtenons des performances bien plus correctes qui sont les suivantes.

```
Cross-validated accuracy: 88.25%
Classification Report:
              precision    recall  f1-score   support

           0       0.94      0.82      0.87       481
           1       0.84      0.95      0.89       481

    accuracy                           0.88       962
   macro avg       0.89      0.88      0.88       962
weighted avg       0.89      0.88      0.88       962
```

Cette fois-ci l'accuracy du modèle est à 88% cependant, nous obtenons des performances relativement satisfaisantes sur les 2 classes.

A présent, avant d'évaluer l'apport individuel de chacune sur les performances du modèle, nous allons énoncer les à priories que nous avions sur les features.

- is_capitalized : Cette feature nous a paru très important quand dans le français écrit, la présence d'une lettre majuscule est un trait caractéristique des noms qui peut réellement être utilisé pour discirminer un mot. Notre à priorie était donc que cette feature allait être importante dans la performance du model.

- length : Pour cette feature, nous étions un peu plus perplexes sur la significativité de son impact. En effet, elle nous semblait être assez utile dans les cas dont la valeur est extrème car les noms sont rarement très petits ou très grands, cependant, à elle seule, elle ne fournissait selon nous pas suffisamment d'information sur les mots de longueur moyenne.

En fin de compte, en ajoutant simplement la features "is_capitalized" (avec rééquilibrage du dataset) nous observons une nette augmentation des performances générale du modèle: 

Zero features :
Cross-validated accuracy: 58.52%
```
   precision    recall  f1-score   support

    0       0.55      0.99      0.70       481
    1       0.95      0.18      0.30       481
```
Avec uniquement la feature "is_capitalized" :
Cross-validated accuracy: 76.20%
```
        precision    recall  f1-score   support

    0       0.71      0.89      0.79       481
    1       0.85      0.64      0.73       481

```


Egalement, avec uniquement la feature "length" nous obtenons également une amélioration significative par rapport à l'absence de feature :

```
Cross-validated accuracy: 73.91%
Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.59      0.69       481
           1       0.68      0.89      0.77       481
```

En somme, il s'est avéré que la feature "is_capitalized" avait comme nous l'avions anticipé un impact significatif mais pour autant, la feature lenght a également un impact très significatif sur les performances que nous n'avions pas autant anticipé.

En somme, nous avons pu observer dans ces multiples tests que l'ajout de ces features était très important pour un bon modeling mais également que dû au déséquilibrage du dataset, nous devions de rééquilibrer pour ne pas créer de modèle biaisé.

Les pipelines train et test ont également été modifiés pour incorporer cette tâche.

## Troubleshooting

Nous avons rencontré une multitude de problèmes relatif à la création d'un bon modèle pour la dernière tâche "find_comic_name" et en particulier pour son utilisation en combinaison avec les résultats des deux autres modèles.

Le problème est l'extrème sous représentatrion des différents noms comparativement aux autres mots. Ainsi, nous avons effectivement crée un modèle utilisant un label encoder mais les résultats de ce modèle sont les suivants :

```
Cross-validated accuracy: 94.49%
   precision    recall  f1-score   support

           0       0.00      0.00      0.00         1
           1       0.00      0.00      0.00         1
           2       0.00      0.00      0.00         1
           3       0.00      0.00      0.00         1
           4       0.00      0.00      0.00         1
           5       0.00      0.00      0.00         1
           6       0.00      0.00      0.00         1
           7       0.00      0.00      0.00         1
           8       0.00      0.00      0.00         3
           9       0.00      0.00      0.00         1
          10       0.00      0.00      0.00         1
          11       0.00      0.00      0.00         3
          12       0.00      0.00      0.00         1
          13       0.00      0.00      0.00         1
          14       0.00      0.00      0.00         2
          15       0.00      0.00      0.00         2
          16       0.00      0.00      0.00         1
          17       0.00      0.00      0.00         1
          18       0.00      0.00      0.00         1
          19       0.00      0.00      0.00         1
          20       0.00      0.00      0.00         1
          21       0.00      0.00      0.00         1
          22       0.00      0.00      0.00         2
          23       0.00      0.00      0.00         1
          24       0.00      0.00      0.00         2
          25       0.00      0.00      0.00         1
          26       1.00      0.89      0.94         9
          27       1.00      1.00      1.00         3
          28       1.00      0.60      0.75         5
          29       0.00      0.00      0.00         1
          30       1.00      1.00      1.00         6
          31       0.00      0.00      0.00         1
          32       0.00      0.00      0.00         1
          33       0.00      0.00      0.00         1
          34       1.00      0.86      0.92         7
          35       0.00      0.00      0.00         1
          36       0.67      0.67      0.67         3
          37       0.00      0.00      0.00         1
          38       0.00      0.00      0.00         2
          39       0.00      0.00      0.00         1
          40       0.00      0.00      0.00         1
          41       1.00      0.71      0.83         7
          42       0.00      0.00      0.00         2
          43       0.00      0.00      0.00         1
          44       1.00      1.00      1.00         7
          45       0.94      1.00      0.97       905

    accuracy                           0.94       999
   macro avg       0.19      0.17      0.18       999
weighted avg       0.90      0.94      0.92       999
```

En effet, malgrès une bonne accuracy globale, il s'avère en réalité que le modèle était très bon sur certains label mais très mauvais sur d'autres.

Ainsi, pour parvenir à une pipeline pertinente combinant les 3 tâches, la solution alternative que nous avions trouvée était de mettre en chaine les 2 premiers modèles :

Si le nom de la vidéo est prédit comme comique, la prédiction du second modèle est lancé pour détecter si les tokens sont des noms. Ainsi si au moins un token est un nom, nous récupérons son index et l'utilisons pour trouver ce mot (find_comic_name). Cette méthode nous permet ainsi de trouve les noms de comic sans devoir faire une ```label_encoder.inverse_transform(...)``` pour retrouver le nom qui a été prédit (à partir du modèle qui est biaisé).


Pour autant, nous avons tout de même intégré ce 3e modèle aux pipelines train et test mais avons surtout crée une dernière pipeline nommé "use" qui prend en argument les différents modèles ainsi qu'un nom de vidéo et qui essaie de trouver les noms de comique après avoir prédit si le titre est comique et si des noms sont présents.

```
Names find by index for the video name: Seins en quête d’une libération - La chronique d'Isabelle Sorente
Isabelle
Sorente
```


## Infos supplémentaires :

Pour mener à bien la suite du projet et notamment la pipeline "use" combinant les différentes tâches, nous avons mis en place la possibilité de dump le vectoriser dans la pipeline "train".
Pour utiliser cette nouvelle pipeline, il est donc nécessaire de "train" un modèle pour les 3 taches et de les importer.