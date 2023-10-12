### Partie 1: Text classification

#### Librairies
- `sklearn.feature_extraction.text`: Cette librairie a plusieurs méthodes pour extraire des features depuis du texte, par exemple `CountVectorizer` ou `TfidfVectorizer`.
- `NLTK`: Excellente pour le prétraitement du texte en anglais, mais nous devons faire attention aux spécificités de la langue française.

#### Tasks
1. **Baseline Model**
   - Définir une métrique d'évaluation (par exemple, la précision)
   - Créer un modèle qui prédit toujours la classe majoritaire pour avoir une baseline à battre.

2. **Expérimentation**
   - Tester différentes features (bag of words, TF-IDF, etc.) et modèles (Logistic Regression, SVM, etc.)
   
3. **Rapport**
   - Documenter les expérimentations, les résultats obtenus, et les leçons apprises.
   
4. **Codebase**
   - Assurer que les scripts pour l'entraînement et la prédiction sont fonctionnels et bien documentés.
   
### Partie 2: Named-entity recognition

#### Tasks
1. **Baseline Model**
   - Pour la NER, nous pourrions utiliser une approche basée sur les règles comme baseline (par exemple, classer tous les mots capitalisés comme noms propres).

2. **Feature Engineering**
   - Des features comme si le mot est capitalisé, la longueur du mot, le mot précédent/suivant, la ponctuation, etc. peuvent être utiles.
   
3. **Rapport**
   - Documenter les résultats et les insights obtenus durant cette étape.

4. **Codebase**
   - Assurer que les scripts pour l'entraînement et la prédiction sont fonctionnels et bien documentés.
   
### Partie 3: Assembler les modèles

#### Tasks
1. **Baseline Model**
   - Nous pourrions utiliser une approche simple comme une règle heuristique ou un modèle basique pour prédire le nom du comique.

2. **Assemblage des Modèles**
   - Cela pourrait être fait en utilisant les prédictions du modèle de classification comme une feature pour le modèle de NER.
   
3. **Rapport**
   - Documenter les résultats, défis, et insights obtenus durant cette étape.

4. **Codebase**
   - Assurer que les scripts pour l'entraînement et la prédiction sont fonctionnels et bien documentés.

### Notes Générales

- **Gestion des Données**: Assurez-vous de bien comprendre vos données et envisagez de faire une analyse exploratoire des données (AED) pour obtenir des insights intéressants qui pourraient vous aider dans l'ingénierie des fonctionnalités.
  
- **Prétraitement du Texte**: Le nettoyage du texte (suppression des stopwords, stemming/lemmatization, etc.) est crucial pour améliorer les performances des modèles.

- **Validation Croisée**: Utilisez la validation croisée pour évaluer les performances des modèles et éviter le surajustement.

- **Optimisation des Hyperparamètres**: N’oubliez pas de régler les hyperparamètres de vos modèles pour améliorer leurs performances.

- **Interprétabilité**: Puisque vous souhaitez utiliser des modèles explicables, pensez à évaluer et à documenter l'importance des features.

Pour la mise en œuvre de chaque étape, nous pourrions approfondir davantage. Puisque l'étendue de chaque étape pourrait être assez vaste, je suggère de nous concentrer sur une étape à la fois. Que diriez-vous de commencer par la première partie concernant la classification de texte? Si vous avez des données spécifiques sur lesquelles vous souhaitez que nous travaillions, n'hésitez pas à les partager et nous pourrons explorer ensemble.


#### Mise en place
##### make_feature :

- TfidfVectorizer est utilisé à la place de CountVectorizer.
- stop_words='english' demande au vectoriseur d'ignorer les mots vides courants en anglais. Vous devrez peut-être ajuster cela pour le français ou fournir votre propre liste de mots vides.
- preprocessor=preprocess_text applique une fonction de prétraitement de texte que nous définissons pour effectuer quelques étapes de nettoyage de texte de base.
- ngram_range=(1, 2) indique au vectoriseur d'utiliser non seulement des mots individuels comme caractéristiques, mais aussi des bigrammes (paires de mots adjacents).



## A rendre
fonction train , test , evaluate
faire un fichier de config pemetteant de tester plusieurs types de features pour un model donné (et optionnellement plusieurs modeles)
ajouter la possibilité de ne pas train sur tout le dataset