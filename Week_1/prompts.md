# Prompts – Week 2 (Data Preparation and Baseline)

Liste de prompts simples qui suivent les étapes du notebook `week2_data_preparation_baseline.ipynb`.

## 1. Comprendre les données

- Charge le fichier `diabetic_data.csv` avec pandas et montre-moi les 5 premières lignes.
- Combien y a-t-il de lignes, de colonnes et de patients différents ?
- Que signifie la colonne `discharge_disposition_id` ? Utilise le fichier `IDS_mapping.csv`.
- Quels codes de sortie correspondent à un décès ou à un hospice ?
- Supprime les séjours où le patient est décédé ou est parti en hospice.

## 2. La cible

- Crée une colonne `target` qui vaut 1 si `readmitted` est `<30`, sinon 0.
- Pourquoi est-ce qu'on met `>30` avec `NO` et pas avec `<30` ?

## 3. Nettoyage

- Remplace les `?` par des valeurs manquantes.
- Transforme la colonne `age` (`[70-80)`) en nombre (le milieu de l'intervalle).
- Regroupe les codes ICD-9 de `diag_1`, `diag_2` et `diag_3` en grandes catégories (circulatoire, respiratoire, diabète…).
- Que veut dire `None` dans `A1Cresult` et `max_glu_serum` ? Remplace-le par `NotMeasured`.
- Les colonnes `admission_type_id` etc. sont des nombres : faut-il les traiter comme des nombres ou comme des catégories ?

## 4. Découpage train / validation / test

- C'est quoi le data leakage ? Donne-moi des exemples dans mon dataset.
- Un même patient apparaît plusieurs fois : comment faire un split sans qu'il soit à la fois dans le train et le test ?
- Fais un split 80/20 par patient avec `GroupShuffleSplit` et `random_state=42`.
- Crée 5 folds de validation croisée groupés par patient avec `StratifiedGroupKFold`.
- Vérifie avec des `assert` qu'aucun patient n'est dans deux ensembles différents.

## 5. Analyse (sur le train uniquement)

- Affiche la répartition de la cible dans le train avec un graphique.
- Est-ce que les classes sont déséquilibrées ? Pourquoi l'accuracy est une mauvaise métrique ici ?
- Affiche le pourcentage de valeurs manquantes par colonne.
- Quelles colonnes faut-il supprimer (trop de valeurs manquantes ou constantes) ?
- Montre le taux de réadmission selon le nombre d'hospitalisations l'année précédente.

## 6. Pipeline de preprocessing

- Pourquoi faut-il mettre l'imputation et la standardisation dans un `Pipeline` ?
- Crée un `ColumnTransformer` : imputation + `StandardScaler` pour les colonnes numériques, imputation + `OneHotEncoder` pour les colonnes catégorielles.
- Comment gérer les catégories rares ou jamais vues avec `OneHotEncoder` ?
- Enregistre le dataset nettoyé dans un fichier CSV.

## 7. Baseline

- C'est quoi un modèle baseline et pourquoi en faut-il un ?
- Entraîne un `DummyClassifier` qui prédit toujours la classe majoritaire.
- Entraîne une régression logistique avec `class_weight="balanced"`.
- À quoi sert `class_weight="balanced"` ?

## 8. Évaluation

- Quelle métrique utiliser quand la classe positive est rare ?
- C'est quoi la différence entre PR-AUC et ROC-AUC ?
- Évalue les deux modèles en validation croisée et donne la moyenne et l'écart-type.
- Calcule la précision et le rappel si on ne signale que les 20 % de patients les plus à risque.
- Trace la courbe précision-rappel, la courbe ROC et la matrice de confusion.
- Comment vérifier qu'il n'y a pas de fuite de données ? (test avec des labels mélangés)
- Est-ce que mon modèle fait de l'overfitting ? Compare le score train et validation.

## 9. Interprétation

- Affiche les 10 coefficients les plus positifs et les 10 plus négatifs de la régression logistique.
- Comment interpréter un coefficient de régression logistique ?
- Pourquoi les catégories rares ont-elles des coefficients très grands ?
- Résume les résultats de la semaine 2 et propose des pistes pour la semaine 3.

# Prompts – Week 3 (Model Development)

Sélection non exhaustive de prompts, reformulés, utilisés pendant l'exécution et la relecture du notebook `week3_model_development.ipynb`.

## 1. Temps de calcul et progression

- La cellule de tuning tourne depuis 50 minutes : pourquoi est-ce si long ?
- Est-ce que cette cellule affiche bien la progression en direct ?
- Dois-je relancer la cellule de tuning après un redémarrage du noyau ?
- Combien de temps vont prendre les sections 4.1, 4.2, 6 et 7 ?
- Vérifie que la première cellule de la section 6 est correcte et affiche sa progression.

## 2. Débogage

- J'ai l'erreur `NameError: name 'itertools' is not defined` : que se passe-t-il ?
- À partir de quelle cellule dois-je relancer ?
- Le tuning de la random forest est fini mais la suite ne se lance pas : pourquoi ?

## 3. Vérification et interprétation

- Vérifie qu'il n'y a pas de data leakage ou d'autre problème dans le notebook.
- Fais-moi un résumé des résultats et dis-moi s'ils sont bons.

## 4. Rédaction

- Mets à jour le fichier de prompts.
