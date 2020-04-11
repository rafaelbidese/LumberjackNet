"""
Reads the database of sound information, forms
a python dictionary, then just prints

Database is a JSON object, with the outermost
object "trees" being an array of other objects,
each one describing a single stem. For each stem
in the database, the following are defined:

"directory", "fileName" - subdirectory under which file named fileName lives
    Each is a .wav file containing 3 seconds of sound covering before,
    during, and after the severing of the stem
"date", "location" - day and approximate location of data collection
    Two sites were visited, multiple visits each. Both were loblolly
    pine plantations. Trees were harvested by two crews, one per
    site, both ran Tigercat feller-bunchers equipped with hotsaws.
"DBH" - stem diameter at breast height, in inches.

"""
import json

trees = None

with open(r'C:\Users\rzb0087\Downloads\d8zvkyf4h2-1\sounds_DBH_inches.json','r') as json_file:
    json_data = json_file.read()
    trees = json.loads(json_data)

print(json.dumps(trees, sort_keys=True, indent=4))
print("Number of trees is %d" % len(trees['trees']))
