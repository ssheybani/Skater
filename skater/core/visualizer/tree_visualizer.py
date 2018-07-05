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
def visualize(estimator, feature_names=None, class_names=None, color_list=None, enable_node_id=True, seed=2):
    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data, filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names,
                    class_names=class_names, node_ids=enable_node_id)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    # if color is not assigned, pick color uniformly random from the color list defined above
    color_names = color_list if color_list is not None else _get_colors(len(class_names), seed)
    print(color_names)
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
