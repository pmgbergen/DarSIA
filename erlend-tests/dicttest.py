f = open("images/metadata/ROI-test.txt")

dicti = {}
for line in f:
    key, value = line.split(":")
    dicti[key] = value
print(type(dicti["Width"]))
