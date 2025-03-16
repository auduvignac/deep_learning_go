# Entraînement d'un réseau de neurones pour jouer au Go

## Description

- [https://www.lamsade.dauphine.fr/~cazenave/DeepLearningProject.html](https://www.lamsade.dauphine.fr/~cazenave/DeepLearningProject.html)  
- L'objectif est d'entraîner un réseau pour jouer au jeu de Go.  
- Afin de garantir une équité en termes de ressources d'entraînement, le nombre de paramètres des réseaux soumis doit être inférieur à 100 000.  
- Le nombre maximal d'étudiants par équipe est de deux.  
- Les données utilisées pour l'entraînement proviennent des parties auto-jouées du programme Katago Go.  
- Le jeu de données d'entraînement contient un total de 1 000 000 de parties différentes.  
- Les données d'entrée sont composées de 31 plans de taille 19x19 :  
  - Couleur au trait  
  - Échelles  
  - État actuel sur deux plans  
  - Deux états précédents sur plusieurs plans  
- Les cibles de sortie sont :  
  - **La politique** : un vecteur de taille 361 avec `1.0` pour le coup joué, `0.0` pour les autres coups.  
  - **La valeur** : une valeur entre `0.0` et `1.0` fournie par la recherche d'arbre Monte-Carlo, représentant la probabilité de victoire de Blanc.

- Le projet a été écrit et fonctionne sous Ubuntu 22.04.  
- Il utilise TensorFlow 2.9 et Keras pour le réseau.  
- Un exemple de réseau convolutionnel avec deux têtes est donné dans le fichier `golois.py` et est sauvegardé dans le fichier `test.h5`.  
- Les réseaux que vous concevez et entraînez doivent également avoir les mêmes têtes de politique et de valeur et être sauvegardés au format `.h5`.  
- Un exemple de réseau et un épisode d'entraînement sont fournis dans le fichier `golois.py`.  
- Si vous souhaitez compiler la bibliothèque Golois, vous devez installer **Pybind11** et exécuter `compile.sh`.

## Tournois

- Toutes les deux semaines environ, j'organiserai un tournoi entre les réseaux que vous téléchargez.  
- Chaque nom de réseau correspond aux noms des étudiants qui ont conçu et entraîné le réseau.  
- Le modèle doit être sauvegardé au format **Keras h5**.  
- Un tournoi en **round robin** sera organisé et les résultats seront envoyés par e-mail.  
- Chaque réseau sera utilisé par un moteur **PUCT**, qui disposera de **2 secondes de temps CPU** par coup pour jouer dans le tournoi.

## Exemple de réseau

```python
planes = 31
moves = 361
N = 10000
epochs = 20
batch = 128
filters = 32
input_data = np.random.randint(2, size=(N, 19, 19, planes))
input_data = input_data.astype ('float32')
policy = np.random.randint(moves, size=(N,))
policy = keras.utils.to_categorical (policy)
value = np.random.randint(2, size=(N,))
value = value.astype ('float32')
end = np.random.randint(2, size=(N, 19, 19, 2))
end = end.astype ('float32')
groups = np.zeros((N, 19, 19, 1))
groups = groups.astype ('float32')

input = keras.Input(shape=(19, 19, planes), name='board')
x = layers.Conv2D(filters, 1, activation='relu', padding='same')(input)
for i in range (5):
  x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)
value_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
value_head = layers.Flatten()(value_head)
value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)
model = keras.Model(inputs=input, outputs=[policy_head, value_head])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9),
loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
loss_weights={'policy' : 1.0, 'value' : 1.0},
metrics={'policy': 'categorical_accuracy', 'value': 'mse'})
for i in range (1, epochs + 1):
  print ('epoch ' + str (i))
  golois.getBatch (input_data, policy, value, end, groups, i * N)
  history = model.fit(input_data,
  {'policy': policy, 'value': value},
  epochs=1, batch_size=batch)
  if (i % 5 == 0):
  gc.collect ()
  if (i % 20 == 0):
  golois.getValidation (input_data, policy, value, end)
  val = model.evaluate (input_data,
  [policy, value], verbose = 0, batch_size=batch)
  print ("val =", val)
  model.save ('test.h5')
```

## Instructions :  
- Entraînez un réseau pour jouer au Go.  
- Soumettez les réseaux entraînés **avant samedi soir**.  
- Tournoi des réseaux **chaque dimanche**.  
- Téléchargez un réseau **avant la fin de la session**.
