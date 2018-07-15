from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
import numpy as np


# reference: http://wingraphviz.sourceforge.net/wingraphviz/language/colorname.htm
color_schemes = ['aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
                 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
                 'cornsilk', 'crimson', 'cyan', 'darkgoldenrod', 'darkgreen', 'darkkhaki', 'darkolivegreen', 'darkorange',
                 'darkorchid', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey',
                 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
                 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green',
                 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender',
                 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrod',
                 'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen',
                 'lightskyblue', 'lightslateblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow',
                 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
                 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
                 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
                 'navyblue', 'oldlace', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
                 'palevioletred', 'papayawhip', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown',
                 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue',
                 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato',
                 'turquoise', 'violet', 'violetred', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']


def _get_colors(num_classes, random_state=1):
    np.random.seed(random_state)
    color_index = np.random.randint(0, len(color_schemes), num_classes)
    colors = np.array(color_schemes)[color_index]
    return colors


# https://stackoverflow.com/questions/48085315/interpreting-graphviz-output-for-decision-tree-regression
# https://stackoverflow.com/questions/42891148/changing-colors-for-decision-tree-plot-created-using-export-graphviz
# Color scheme info: http://wingraphviz.sourceforge.net/wingraphviz/language/colorname.htm
# Currently, supported only for sklearn models
def plot_tree(estimator, feature_names=None, class_names=None, color_list=None, enable_node_id=True, seed=2):
    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data, filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names,
                    class_names=class_names, node_ids=enable_node_id)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    # if color is not assigned, pick color uniformly random from the color list defined above
    color_names = color_list if color_list is not None else _get_colors(len(class_names), seed)
    default_color = 'cornsilk'

    # Query for the node list to change properties
    nodes = graph.get_node_list()
    for node in nodes:
        if node.get_name() not in ('node', 'edge'):
            values = estimator.tree_.value[int(node.get_name())][0]
            # 1. Color only the leaf nodes, One way to identify leaf nodes is to check on the values which
            #    should represent a distribution only for one class
            # 2. mixed nodes get the default color
            node.set_fillcolor(color_names[np.argmax(values)]) if max(values) == sum(values) \
                else node.set_fillcolor(default_color)

    # Query for the edge list to change properties
    edges = graph.get_edge_list()
    for ed in edges:
        ed.set_color('steelblue')
    return graph


return_value = lambda estimator_type, v: 'Predicted Label: {}'.format(str(np.argmax(v))) if 'classifier' \
    else 'Output: {}'.format(str(v))


# Current implementation is specific to sklearn models.
# Reference: https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
# TODO: Figure out ways to make it generic for other frameworks
def tree_to_text(tree, feature_names, estimator_type='classifier'):
    # definining colors
    label_value_color = "\033[1;34;49m"  # blue
    split_criteria_color = "\033[0;32;49m"  # green
    if_else_quotes_color = "\033[0;30;49m"  # if and else quotes

    left_node = tree.tree_.children_left
    right_node = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features_names = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    # Reference: https://github.com/scikit-learn/scikit-learn/blob/a24c8b464d094d2c468a16ea9f8bf8d42d949f84/sklearn/tree/_tree.pyx
    TREE_LEAF = -1
    TREE_UNDEFINED = -2

    # define "if and else" string patterns for extracting the decision rules
    if_str_pattern = lambda offset, s_c, f_n, node, ie_c: offset + "if {}{}".format(s_c, f_n[node]) + " <= {}"\
        .format(str(s_c[node])) + ie_c + " {"

    other_str_pattern = lambda offset, color_val, str_type: offset + color_val + str_type

    def recurse_tree(left_node, right_node, split_criterias, features_names, node, depth=0):
        offset = "  " * depth
        if threshold[node] != TREE_UNDEFINED:
            print(if_str_pattern(offset, split_criteria_color, features_names, node, if_else_quotes_color))
            if left_node[node] != TREE_LEAF:
                recurse_tree(left_node, right_node, threshold, features_names, left_node[node], depth + 1)
                print(other_str_pattern(offset, if_else_quotes_color, "} else {"))
                if right_node[node] != TREE_LEAF:
                    recurse_tree(left_node, right_node, threshold, features_names, right_node[node], depth + 1)
                print(other_str_pattern(offset, if_else_quotes_color, "}"))
        else:
            print(offset, label_value_color, return_value(estimator_type, value[node]))

    recurse_tree(left_node, right_node, threshold, features_names, 0)
