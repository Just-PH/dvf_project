import os
from typing import Optional, Union, Tuple, List, Any, Dict, Iterator
import joblib
import numpy as np
import shap
import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import StackingRegressor
from pygam import LinearGAM, s
# from model_script.script_XGBoost_model_polars import DataCleaner, FeatureCreator, AnomalyFilter, WeightedPOICountsTransformer, data_loader, build_xgboost_model,save_model,plot_train_test_predictions

class GAMRegressor(BaseEstimator, RegressorMixin):
    """
    Regressor based on a Generalized Additive Model (GAM).

    This estimator encapsulates a LinearGAM and allows it to be used
    within a scikit-learn pipeline.

    Attributes:
        term: Term(s) defining the structure of the GAM.
        max_iter: Maximum number of iterations for fitting.
        lam: Regularization parameter.
        gam_: Fitted instance of LinearGAM.
    """
    def __init__(self, term: Any, max_iter: int = 100, lam: float = 0.5) -> None:
        self.term = term
        self.max_iter = max_iter
        self.lam = lam
        self.gam_: Optional[LinearGAM] = None  # Will be initialized in fit()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GAMRegressor':
        """Fit the LinearGAM model on the data X and target y."""
        self.gam_ = LinearGAM(self.term, max_iter=self.max_iter, lam=self.lam).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predictions of the fitted model for X."""
        return self.gam_.predict(X)

    def get_params(self,  deep: bool = True) -> Dict[str, Any]:
        """Return the parameters of the estimator."""
        return {"term": self.term, "max_iter": self.max_iter, "lam": self.lam}

    def set_params(self, **params: Any) -> 'GAMRegressor':
        """Set the parameters of the estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class CombinedXGBModel(BaseEstimator, RegressorMixin):
    """
    Combined model integrating physical and contextual models.

    Depending on the 'gam' parameter, this model uses either GAMs (via GAMRegressor)
    or standard models (ElasticNet for the physical part and XGBRegressor for the contextual part).
    The stacking option allows combining the predictions via a StackingRegressor.

    Parameters:
        physical_params (dict): Parameters for the physical model. To use GAM, must contain key 'n_features'.
        contextual_params (dict): Parameters for the contextual model. To use GAM, must contain key 'n_features'.
        physical_weight (float): Weight of the physical model in the final combination (default 0.5).
        save (bool): Whether to save the models.
        stacking (bool): Enables stacking if True.
        gam (bool): Uses GAMs for the models if True.
    """
    def __init__(self,
                 physical_params: Dict[str, Any],
                 contextual_params: Dict[str, Any],
                 physical_weight: float = 0.5,
                 save: bool = False,
                 stacking: bool = False,
                 gam: bool = False) -> None:
        self.physical_params = physical_params
        self.contextual_params = contextual_params
        self.physical_weight = physical_weight
        self.contextual_weight = 1 - physical_weight
        self.save = save
        self.stacking = stacking
        self.gam = gam
        self._build_models()

    def _build_models(self)-> None:
        """
        Build internal models based on chosen options (GAM or standard models).
        """
        if self.gam:
            # Retrieve number of features for each group
            n_phys = self.physical_params.get('n_features')
            n_ctx = self.contextual_params.get('n_features')
            if n_phys is None or n_ctx is None:
                raise ValueError("To use GAM, specify 'n_features' in both physical_params and contextual_params.")
            # Build terms for the physical GAM
            terms_phys = s(0)
            for i in range(1, n_phys):
                terms_phys += s(i)
            self.physical_model = GAMRegressor(
                term=terms_phys,
                max_iter=self.physical_params.get('max_iter', 100),
                lam=self.physical_params.get('lam', 0.5)
            )
            # Build terms for the contextual GAM
            terms_ctx = s(0)
            for i in range(1, n_ctx):
                terms_ctx += s(i)
            self.contextual_model = GAMRegressor(
                term=terms_ctx,
                max_iter=self.contextual_params.get('max_iter', 100),
                lam=self.contextual_params.get('lam', 0.3)
            )
            if self.stacking:
                # In stacking, the final estimator receives 2 features (the predictions of the two base models)
                self.stacking_model = StackingRegressor(
                    estimators=[('physical', self.physical_model), ('contextual', self.contextual_model)],
                    final_estimator=ElasticNet(fit_intercept=False),
                    passthrough=True
                )
        else:
            # Use standard models
            self.physical_model = ElasticNet(**self.physical_params)
            self.contextual_model = xgb.XGBRegressor(**self.contextual_params)
            if self.stacking:
                self.stacking_model = StackingRegressor(
                    estimators=[('physical', self.physical_model), ('contextual', self.contextual_model)],
                    final_estimator=ElasticNet(fit_intercept=False),
                    passthrough=True
                )

    def fit(self,
            X: Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]], np.ndarray],
            y: np.ndarray) -> 'CombinedXGBModel':
        """
        Fit the models on the data.

        X can be a CustomTupleWrapper, a list or an array of tuples (X_physical, X_contextual),
        or a tuple directly.
        """
        if isinstance(X, CustomTupleWrapper):
            X_physical, X_contextual = X.X_physical, X.X_contextual
        elif isinstance(X, (list, np.ndarray)):
            sample = X[0]
            if isinstance(sample, (tuple, list)) and len(sample) == 2:
                X_physical, X_contextual = map(np.array, zip(*X))
            else:
                raise ValueError("X is not in the expected format (list of 2-element tuples).")
        else:
            X_physical, X_contextual = X

        if self.stacking:
            X_combined = np.hstack((X_physical, X_contextual))
            self.stacking_model.fit(X_combined, y)
        else:
            self.physical_model.fit(X_physical, y)
            self.contextual_model.fit(X_contextual, y)
        return self

    def predict(self,
                X: Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]
               ) -> np.ndarray:
        """
        Return the prediction of the combined model based on data X.
        """
        if isinstance(X, CustomTupleWrapper):
            X_physical, X_contextual = X.X_physical, X.X_contextual
        elif isinstance(X, (list, np.ndarray)):
            sample = X[0]
            if isinstance(sample, (tuple, list)) and len(sample) == 2:
                X_physical, X_contextual = map(np.array, zip(*X))
            else:
                raise ValueError("X is not in the expected format (list of 2-element tuples).")
        else:
            X_physical, X_contextual = X

        if self.stacking:
            X_combined = np.hstack((X_physical, X_contextual))
            return self.stacking_model.predict(X_combined)
        else:
            pred_physical = self.physical_model.predict(X_physical)
            pred_contextual = self.contextual_model.predict(X_contextual)
            combined_preds = (self.physical_weight * pred_physical +
                              self.contextual_weight * pred_contextual)
            return combined_preds

    def save_models(self, directory: str = 'saved_models') -> None:
        """
        Save the models to the specified directory.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.stacking:
            joblib.dump(self.stacking_model, os.path.join(directory, 'stacking_model.pkl'))
        else:
            joblib.dump(self.physical_model, os.path.join(directory, 'physical_model.pkl'))
            joblib.dump(self.contextual_model, os.path.join(directory, 'contextual_model.pkl'))

    def load_models(self, directory: str = 'saved_models') -> None:
        """
        Load the models from the specified directory.
        """
        if self.stacking:
            self.stacking_model = joblib.load(os.path.join(directory, 'stacking_model.pkl'))
        else:
            self.physical_model = joblib.load(os.path.join(directory, 'physical_model.pkl'))
            self.contextual_model = joblib.load(os.path.join(directory, 'contextual_model.pkl'))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Return the parameters of the combined model.

        If deep is True, the parameters of the sub-models are also included.
        """
        params = {
            'physical_params': self.physical_params,
            'contextual_params': self.contextual_params,
            'physical_weight': self.physical_weight,
            'save': self.save,
            'stacking': self.stacking,
            'gam': self.gam
        }
        if deep:
            for key, val in self.physical_params.items():
                params[f'physical_params__{key}'] = val
            for key, val in self.contextual_params.items():
                params[f'contextual_params__{key}'] = val
        return params

    def set_params(self, **params: Any) -> 'CombinedXGBModel':
        """
        Set the parameters of the combined model and rebuild internal models.
        """
        for key, value in params.items():
            if key.startswith("physical_params__"):
                subkey = key.split("__", 1)[1]
                self.physical_params[subkey] = value
            elif key.startswith("contextual_params__"):
                subkey = key.split("__", 1)[1]
                self.contextual_params[subkey] = value
            else:
                setattr(self, key, value)
        self._build_models()
        return self


class CustomGridSearchCV(GridSearchCV):
    """
    Custom variant of GridSearchCV to handle tuples of matrices.

    If X is a tuple, it is wrapped in a CustomTupleWrapper to avoid validation errors.
    """
    def __init__(self, estimator: Any, param_grid: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(estimator, param_grid, **kwargs)

    def fit(self,
            X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]],
            y: Optional[np.ndarray] = None,
            **fit_params: Any
        ) -> GridSearchCV:
        """Fit the model, wrapping X if necessary."""
        if isinstance(X, tuple):
            X = CustomTupleWrapper(X)
        else:
            X = self._split_features(X)
        return super().fit(X, y, **fit_params)

    def predict(self,
                X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]
        ) -> np.ndarray:

        """Return predictions, wrapping X if necessary."""
        if isinstance(X, tuple):
            X = CustomTupleWrapper(X)
        else:
            X = self._split_features(X)
        return super().predict(X)

    def _split_features(self, X: np.ndarray) -> 'CustomTupleWrapper':
        """
        Split X into two matrices (physical and contextual) by dividing in half.
        Returns a CustomTupleWrapper.
        """
        X_physical = X[:, :X.shape[1] // 2]
        X_contextual = X[:, X.shape[1] // 2:]
        return CustomTupleWrapper((X_physical, X_contextual))


class CustomTupleWrapper:
    """
    Encapsulates a tuple (X_physical, X_contextual) and behaves like a sequence of samples.

    For each index i, __getitem__(i) returns (X_physical[i], X_contextual[i]).
    """
    def __init__(self, X_tuple: Tuple[np.ndarray, np.ndarray]) -> None:
        self.X_physical, self.X_contextual = X_tuple

    def __getitem__(self, index: Union[int, slice]) -> Union[Tuple[np.ndarray, np.ndarray], 'CustomTupleWrapper']:
        """
        Return the tuple for the sample at the given index.
        If index is a slice, return a new instance with the corresponding slice.
        """
        if isinstance(index, slice):
            return CustomTupleWrapper((self.X_physical[index], self.X_contextual[index]))
        else:
            return (self.X_physical[index], self.X_contextual[index])

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.X_physical)

    def __iter__(self)-> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over the samples."""
        for i in range(len(self)):
            yield self[i]


def train_combined_model(
        df_features: pl.DataFrame,
        df_target: pl.DataFrame,
        physical_pipeline: Pipeline,
        physical_features: List[str],
        params_physical: Dict[str, Any],
        contextual_pipeline: Pipeline,
        contextual_features: List[str],
        params_contextual: Dict[str, Any],
        save: bool = False,
        stacking: bool = False,
        gam: bool = False
    ) -> 'CombinedXGBModel':
    """
    Train a combined model using physical and contextual feature data.

    Parameters:
        df_features: DataFrame containing the features.
        df_target: DataFrame containing the target.
        physical_pipeline: Transformation pipeline for physical features.
        physical_features: List of columns for physical features.
        params_physical: Dictionary of parameters for the physical model.
        contextual_pipeline: Transformation pipeline for contextual features.
        contextual_features: List of columns for contextual features.
        params_contextual: Dictionary of parameters for the contextual model.
        save (bool): Whether to save the models.
        stacking (bool): Enables stacking if True.
        gam (bool): Uses GAM if True.

    Returns:
        combined_model: Trained instance of the combined model.
    """
    # Data preparation
    X_physical = physical_pipeline.fit_transform(df_features.select(physical_features).collect())
    X_contextual = contextual_pipeline.fit_transform(df_features.select(contextual_features).collect())
    X_combined = (X_physical, X_contextual)
    y_train = df_target.collect().to_numpy().ravel()

    # Train the combined model
    combined_model = CombinedXGBModel(
        physical_params=params_physical,
        contextual_params=params_contextual,
        stacking=stacking,
        gam=gam
    )
    combined_model.fit(X_combined, y_train)

    if save:
        combined_model.save_models()

    return combined_model


def combined_prediction(
        model: 'CombinedXGBModel',
        df_features: pl.DataFrame,
        physical_pipeline: Pipeline,
        physical_features: List[str],
        contextual_pipeline: Pipeline,
        contextual_features: List[str]
    ) -> np.ndarray:
    """
    Make a combined prediction from physical and contextual feature data.

    Parameters:
        model: Trained combined model.
        df_features: DataFrame containing the features.
        physical_pipeline: Pipeline for physical features.
        physical_features: List of columns for physical features.
        contextual_pipeline: Pipeline for contextual features.
        contextual_features: List of columns for contextual features.

    Returns:
        Predictions from the combined model.
    """
    X_physical = physical_pipeline.transform(df_features.select(physical_features).collect())
    X_contextual = contextual_pipeline.transform(df_features.select(contextual_features).collect())
    X_combined = (X_physical, X_contextual)
    return model.predict(X_combined)


def grid_search_combined_model(
        df_features: pl.DataFrame,
        df_target: pl.DataFrame,
        physical_pipeline: Pipeline,
        physical_features: List[str],
        params_physical: Dict[str, Any],
        contextual_pipeline: Pipeline,
        contextual_features: List[str],
        params_contextual: Dict[str, Any],
        stacking: bool = False,
        gam: bool = False,
        save: bool = False
    ) -> Tuple['CombinedXGBModel', Dict[str, Any]]:
    """
    Perform hyperparameter tuning on the combined model using a custom GridSearchCV.

    Parameters:
        df_features: DataFrame containing the features.
        df_target: DataFrame containing the target.
        physical_pipeline: Pipeline for physical features.
        physical_features: List of columns for physical features.
        params_physical: Dictionary of parameters for the physical model.
        contextual_pipeline: Pipeline for contextual features.
        contextual_features: List of columns for contextual features.
        params_contextual: Dictionary of parameters for the contextual model.
        stacking (bool): Enables stacking if True.
        gam (bool): Uses GAM if True.
        save (bool): Whether to save the best model.

    Returns:
        best_model: The best combined model found.
        best_params: Dictionary of the best parameters.
    """
    # Data preparation
    X_physical = physical_pipeline.fit_transform(df_features.select(physical_features).collect())
    X_contextual = contextual_pipeline.fit_transform(df_features.select(contextual_features).collect())
    X_combined = (X_physical, X_contextual)
    y_train = df_target.collect().to_numpy().ravel()

    # Define the combined model
    combined_model = CombinedXGBModel(
        physical_params=params_physical,
        contextual_params=params_contextual,
        stacking=stacking,
        gam=gam
    )

    # Define the hyperparameter grid
    if gam:
        param_grid = {
            'physical_params__n_splines': [10, 20, 30],
            'physical_params__lam': [0.1, 1, 10],
            'physical_params__spline_order': [2, 3],
            'contextual_params__n_splines': [10, 20, 30],
            'contextual_params__lam': [0.1, 1, 10],
            'contextual_params__spline_order': [2, 3]
        }
    else:
        param_grid = {
            'physical_weight': [0.1, 0.3, 0.5, 0.7, 0.9],
            'physical_params__n_estimators': [100, 200],
            'physical_params__max_depth': [6, 8, 10],
            'physical_params__learning_rate': [0.1, 0.3, 0.5],
            'contextual_params__n_estimators': [100, 200],
            'contextual_params__max_depth': [4, 6, 8],
            'contextual_params__learning_rate': [0.1, 0.3, 0.5]
        }

    # Hyperparameter search with custom GridSearchCV
    grid_search = CustomGridSearchCV(estimator=combined_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_combined, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    if save:
        best_model.save_models()

    return best_model, best_params


def predict_wrapper(
        X: np.ndarray,
        combined_model: 'CombinedXGBModel',
        len_physical: int = 0
    ) -> np.ndarray:
    """
    Split the combined feature array into two parts (physical and contextual)
    and return the prediction of the combined model.

    Parameters:
        X (2D array): Combined feature matrix.
        combined_model: Instance of the combined model.
        len_physical (int): Number of columns corresponding to physical features.

    Returns:
        Prediction from the combined model.
    """
    X_physical = X[:, :len_physical]
    X_contextual = X[:, len_physical:]
    return combined_model.predict((X_physical, X_contextual))


def compute_shap_impact(
        model: 'CombinedXGBModel',
        X: pl.LazyFrame,
        physical_pipeline: Pipeline,
        physical_features: List[str],
        contextual_pipeline: Pipeline,
        contextual_features: List[str],
        sample_size: Optional[int] = None
    ) -> Dict[str, float]:
    """
    Compute the mean impact (percentage) of the physical and contextual models on the meta-model using SHAP.

    Parameters:
        model: Instance of CombinedXGBModel (with stacking enabled).
        X: Input data as a Polars LazyFrame.
        physical_pipeline: Transformation pipeline for physical features.
        physical_features: List of columns for physical features.
        contextual_pipeline: Transformation pipeline for contextual features.
        contextual_features: List of columns for contextual features.
        sample_size (int, optional): Sample size to use for SHAP computation.

    Returns:
        impact_dict (dict): Dictionary with the percentage impact of the physical and contextual models.
    """
    if not model.stacking:
        raise ValueError("SHAP computation is only applicable when stacking is enabled.")

    # Transform the data using the pipelines
    X_physical = physical_pipeline.fit_transform(X.select(physical_features).collect())
    X_contextual = contextual_pipeline.fit_transform(X.select(contextual_features).collect())
    len_physical = X_physical.shape[1]

    # Combine the data into a single 2D array by horizontal concatenation
    X_combined = np.hstack((X_physical, X_contextual))
    if sample_size and sample_size < X_combined.shape[0]:
        indices = np.random.choice(X_combined.shape[0], size=sample_size, replace=False)
        X_sample = X_combined[indices]
    else:
        X_sample = X_combined

    # Define a callable prediction function for SHAP that splits the array appropriately
    wrapped_predict = lambda X: predict_wrapper(X, combined_model=model, len_physical=len_physical)

    # Create the SHAP explainer using wrapped_predict and the background data X_sample
    explainer = shap.Explainer(wrapped_predict, X_sample)
    shap_values = explainer(X_sample)

    # Compute the mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    # Separate the impact between physical and contextual features
    physical_impact = mean_abs_shap[:len_physical].sum()
    contextual_impact = mean_abs_shap[len_physical:].sum()

    total_impact = physical_impact + contextual_impact
    impact_dict = {
        'physical_model_impact': (physical_impact / total_impact) * 100,
        'contextual_model_impact': (contextual_impact / total_impact) * 100
    }

    return impact_dict
