Tutorials
=========

Below are some examples of how you might use this package.

Simple Train and Test
--------------------

Suppose you have a dataset with 5 records and 4 features.
You also have a count of how many times an entity (NER) have been mentioned in each record (fifth column).

You want to train your VTT model and then test it against unseen data. Your unseen data has only 4 records.

.. code-block:: python

  from vtt import VTT

  # Train dataset
  X_train = [[1, 1, 0, 0, 12],
             [1, 1, 0, 0, 14],
             [1, 1, 0, 0, 11],
             [0, 0, 1, 1, 6 ],
             [0, 0, 1, 1, 8 ]]
  y_train = [1, 1, 1, 0, 0]

  # Test datase
  X_test = [[1, 0, 0, 0, 11],
            [1, 0, 0, 0, 12],
            [0, 0, 0, 1, 7 ],
            [0, 0, 0, 1, 5 ]]
  y_test = [1, 1, 0, 0] # Gold Standard

  # Init classifier
  classifier = VTT() 

  # Set the NER collumn (4) and bias/intercept (10). Mind that column count starts at 0
  classifier.set_params(b_4=10)

  # Fit Linear Model
  classifier.fit(X_train, y_train)

  # Predict new instances
  y_predict = classifier.predict(X_test)

