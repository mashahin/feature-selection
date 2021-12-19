# import liberaries
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE, SelectKBest, SelectPercentile
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
# pd.options.display.max_rows = 20

# load data
cancer_data = load_breast_cancer(as_frame=True)
X = cancer_data['data']
y = cancer_data['target']
#print(f"Features shape: {X.shape}")
#print(f"Target data shape: {y.shape}")

# split data into train & test databses
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
#print(f"Traing data shape: {X_train.shape}, {y_train.shape}")
#print(f"Test data shape: {X_test.shape}, {y_test.shape} \n")

# helper functions


def compare_feature_sets(set1, set2):
    if(set1.sort_values().equals(set2.sort_values())):
        print("\tTwo methods selected same features")
    else:
        #print("Two methods selected different features")
        print(
            f"\tTwo selection methods differed by {len([var for var in kbest_features if var not in rfe_features])} features")


def get_roc_auc(model, X, y):
    y_probability = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_probability)


# Baseline with all features
model0 = GradientBoostingClassifier(random_state=42)
model0.fit(X_train, y_train)

print(f"\n======== Baseline Model with {X_train.shape[1]} Features ========")
print(f"Training ROC-AUC: {get_roc_auc(model0, X_train, y_train):.4f} ")
print(f"Test ROC-AUC: {get_roc_auc(model0, X_test, y_test):.4f} ")

# feature importance
features = pd.DataFrame(model0.feature_importances_,
                        index=X.columns, columns=['importance'])

# select imortant features above threshold importance
threshold = 0.01
important_features = features[features['importance'] > threshold].index
print(
    f"\n======== {len(important_features)} Important Features Selected ========")
#print(f"{', '.join(important_features)}")

# model with selected features
model1 = GradientBoostingClassifier(random_state=42)
model1.fit(X_train[important_features], y_train)

print(
    f"Training data ROC AUC: {get_roc_auc(model1, X_train[important_features], y_train):.4f}")
print(
    f"Test data ROC AUC: {get_roc_auc(model1, X_test[important_features], y_test):.4f}")

# selection with recursive feature elimination
rfe = RFE(GradientBoostingClassifier(random_state=42), n_features_to_select=10)
rfe.fit(X_train, y_train)

rfe_features = X_train.columns[rfe.support_]
print(
    f"\n========== {len(rfe_features)} Features Selected with RFE ========")
#print(f"{', '.join(rfe_features)}")

# model for FRE selected features
model2 = GradientBoostingClassifier(random_state=42)
model2.fit(X_train[rfe_features], y_train)

print(
    f"Training ROC AUC (RFE): {get_roc_auc(model2, X_train[rfe_features], y_train):.4f}")
print(
    f"Test ROC AUC (RFE): {get_roc_auc(model2, X_test[rfe_features], y_test):.4f}")

# chech if rfe_feaures are the same as important_features
# important_features.sort_values().equals(rfe_features.sort_values())
print("Feature importance vs RFE")
compare_feature_sets(important_features, rfe_features)


# select KBest features
kbest = SelectKBest(k=10)
kbest.fit(X_train, y_train)

kbest_features = X_train.columns[kbest.get_support()]
print(
    f"\n========== {len(kbest_features)} Features Selected with KBest =========")
#print(f"{', '.join(kbest_features)}")

# model with kbest features
model3 = GradientBoostingClassifier(random_state=42)
model3.fit(X_train[kbest_features], y_train)

print(
    f"Training ROC AUC: {get_roc_auc(model3, X_train[kbest_features], y_train):.4f}")
print(
    f"Test ROC AUC: {get_roc_auc(model3, X_test[kbest_features], y_test):.4f}")

# check for features not selected previously
# [var for var in kbest_features if var not in rfe_features]
print("KBest vs RFE")
compare_feature_sets(kbest_features, rfe_features)

# select with percentile
percentile = SelectPercentile(percentile=33)
percentile.fit(X_train, y_train)

percentile_features = X_train.columns[percentile.get_support()]
print(
    f"\n======== {len(percentile_features)} Percentile Features Selected =======")
#print(f"{', '.join(percentile_features)}")

# model with selected features
model4 = GradientBoostingClassifier(random_state=42)
model4.fit(X_train[percentile_features], y_train)

print(
    f"Training ROC AUC: {get_roc_auc(model4, X_train[percentile_features], y_train):.4f}")
print(
    f"Test ROC AUC: {get_roc_auc(model4, X_test[percentile_features], y_test):.4f}")

# check if features are the same as with KBest
# [percentile_features.sort_values().equals(kbest_features.sort_values())]
print("Percentile vs KBest")
compare_feature_sets(percentile_features, kbest_features)

# combine multiple approaches
selection = pd.DataFrame(index=X.columns)
selection['imp'] = [var in important_features for var in X.columns]
selection['rfe'] = rfe.support_
selection['kbest'] = kbest.get_support()
selection['percentile'] = percentile.get_support()
selection['sum'] = selection.sum(axis=1)
selection.sort_values('sum', ascending=False, inplace=True)

# ckeck the distribution of the sum column
pd.concat([selection['sum'].value_counts(normalize=True),
           selection['sum'].value_counts()], axis=1,
          keys=['prop', 'count'])

# select from combined approaches
selected_features = selection[selection['sum'] > 0].index
print(
    f"\n======== {len(selected_features)} Comnined Features Selected ========")
#print(f"{', '.join(selected_features)}")

# model with cobined features
model5 = GradientBoostingClassifier(random_state=42)
model5.fit(X_train[selected_features], y_train)
print(
    f"Training ROC-AUC: {get_roc_auc(model5, X_train[selected_features], y_train):.4f}")
print(
    f"Test ROC-AUC: {get_roc_auc(model5, X_test[selected_features], y_test):.4f}")
