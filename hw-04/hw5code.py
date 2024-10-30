import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator

def find_best_split(feature_vector, target_vector):
    unique_values = np.unique(feature_vector)

    if len(unique_values) < 2:
        return np.array([]), np.array([]), None, -1

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2

    ginis = np.zeros(thresholds.shape)

    for i, thres in enumerate(thresholds):
        left_mask = feature_vector < thres
        right_mask = ~left_mask

        left_counts = Counter(target_vector[left_mask])
        right_counts = Counter(target_vector[right_mask])

        left_size = left_mask.sum()
        right_size = right_mask.sum()

        p_left_0 = left_counts[0] / left_size if left_size > 0 else 0
        p_left_1 = left_counts[1] / left_size if left_size > 0 else 0

        p_right_0 = right_counts[0] / right_size if right_size > 0 else 0
        p_right_1 = right_counts[1] / right_size if right_size > 0 else 0

        H_left = 1 - (p_left_0 ** 2 + p_left_1 ** 2)
        H_right = 1 - (p_right_0 ** 2 + p_right_1 ** 2)

        ginis[i] = -(left_size / target_vector.shape[0]) * H_left - (right_size / target_vector.shape[0]) * H_right

    gini_best_index = np.argmax(ginis)

    thresholds_best = thresholds[gini_best_index]
    gini_best_value = ginis[gini_best_index]

    return thresholds, ginis, thresholds_best, gini_best_value


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    current_click = clicks[key] if key in clicks else 0
                    ratio[key] = current_click / current_count if current_count > 0 else 0
                sorted_categories = sorted(ratio, key=ratio.get)
                categories_map = {k: i for i, k in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError("Unknown feature type")

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [k for k, v in categories_map.items() if v < threshold]

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError("Unknown feature type")

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"])

    def _predict_node(self, x, node):

        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature] == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Unknown feature type")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])
