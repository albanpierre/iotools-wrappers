
# Loading:
from iotools.xmlio import load_xml
data = load_xml("filename.xml")


# Saving:
from iotools.xmlio import save_xml
from collections import OrderedDict
data = OrderedDict([("main", OrderedDict([("some", "0"), ("xml", "1")]))])
save_xml("filename.xml", data)
