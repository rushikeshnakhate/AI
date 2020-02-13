training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

header = ['color', 'diameter', 'label']


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if isinstance(val, int):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if isinstance(self.value, int):
            condition = ">="
        return "is %s %s %s" % (header[self.column], condition, str(self.value))


def partition(rows, obj):
    true_rows, false_rows = [], []
    for row in rows:
        if obj.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(data):
    counts = class_counts(data)
    impurity = 1
    for lbl in counts:
        print(lbl)
        print(counts[lbl])
        print(float(len(training_data)))
        prob_of_lbl = counts[lbl] / float(len(training_data))
        print(prob_of_lbl)


# Making a map using the folium module
import folium

phone_map = folium.Map()

# Top three smart phone companies by market share in 2016
companies = [
    {'loc': [37.4970, 127.0266], 'label': 'Samsung: ...%'},
    {'loc': [37.3318, -122.0311], 'label': 'Apple: ...%'},
    {'loc': [22.5431, 114.0579], 'label': 'Huawei: ...%'}]

# Adding markers to the map
for company in companies:
    marker = folium.Marker(location=company['loc'], popup=company['label'])
    marker.add_to(phone_map)

# The last object in the cell always gets shown in the notebook
phone_map
