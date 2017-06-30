from sklearn.model_selection import GridSearchCV
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier


class Keras:
    def __init__(self, model_architecture, loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']):
        if model_architecture:
            self.model_architecture = model_architecture
            self.loss = loss
            self.optimizer = optimizer
            self.metrics = metrics
            self.grid = None

    def _create_model(self, optimizer, activation):
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation=activation))
        model.add(Dense(1, activation=activation))
        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        return model

    def train(self):
        pass

    def grid_search(self, x, y, param_grid, n_jobs=-1):
        model = KerasClassifier(build_fn=self._create_model, verbose=0)
        self.grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=n_jobs)
        grid_results = self.grid.fit(x, y)
        return {
            'best_score': grid_results.best_score_,
            'best_params': grid_results.best_params_,
            'grid_results_cv': grid_results.cv_results_
        }

    def randomized_search(self):
        pass
