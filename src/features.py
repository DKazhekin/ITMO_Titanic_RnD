"""Train-only age imputation for a single age-by-sex interaction."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class AgeMaleInteraction(TransformerMixin, BaseEstimator):
    """Multiply raw age in years by the male indicator after median imputation."""

    def fit(self, X, y=None):
        ages = X["Age"].to_numpy(dtype=float)
        if not np.isfinite(ages[~np.isnan(ages)]).all() or np.isnan(ages).all():
            raise ValueError("Age needs at least one finite training observation")
        self.age_median_ = float(np.nanmedian(ages))
        self.n_features_in_ = 2
        self.feature_names_in_ = np.array(["Age", "Sex"], dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "age_median_")
        ages = X["Age"].to_numpy(dtype=float)
        ages = np.where(np.isnan(ages), self.age_median_, ages)
        if not np.isfinite(ages).all() or not X["Sex"].isin(["male", "female"]).all():
            raise ValueError("Interaction requires finite ages and known binary sex values")
        return np.where(X["Sex"].to_numpy() == "male", ages, 0.0).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["AgeMale"], dtype=object)
