# Feature Selection in Scikit-learn

When building a supervised machine learning model, we gather features that are potentially useful for predicting the outcome. Not all features are going to end up being helpful in building the model. It’s often practical to filter out non-predictive features and keep the model lean so that the model is faster, easier to explain to stakeholders and simpler to productionise.

Code in feature-selection.py compares model performance for features selected with different methods in Scikit-learn. It demonstrates a few simple ways to weed out features that are not useful in predicting the outcome and select the features that contribute more.
