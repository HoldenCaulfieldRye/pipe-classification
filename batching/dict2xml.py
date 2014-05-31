from xml.dom.minidom import Document
import copy


class dict2xml(object): # bit.ly/1u0zMeR
    doc     = Document()
    # dict2xml has an attribute, doc, of type Document.
    # Document class: bit.ly/1u0EOYR

    # structure is the arg passed to initialise a dict2xml object.
    # it is the dict (but can also be a list).
    def __init__(self, structure):
        # print '__init__ called with structure:', structure
        if len(structure) == 1:
            rootName    = str(structure.keys()[0])
            print 'want to create new attrib %s'%(rootName)
            self.root   = self.doc.createElement(rootName)
            self.doc.appendChild(self.root)
            self.build(self.root, structure[rootName])
        else:
          print 'weird, len(structure)==%i so not doing anything'%(len(structure))

    def build(self, father, structure):
        # print 'build called with father: %s, structure: %s'%(father, structure)
        if type(structure) == dict:
            for k in structure:
                tag = self.doc.createElement(k)
                father.appendChild(tag)
                self.build(tag, structure[k])

        elif type(structure) == list:
            grandFather = father.parentNode
            tagName     = father.tagName
            grandFather.removeChild(father)
            for l in structure:
                tag = self.doc.createElement(tagName)
                self.build(tag, l)
                grandFather.appendChild(tag)

        else:
            data    = str(structure)
            tag     = self.doc.createTextNode(data)
            father.appendChild(tag)

    def display(self):
        print self.doc.toprettyxml(indent="  ")

if __name__ == '__main__':
    example = {'auftrag':{"kommiauftragsnr":2103839, "anliefertermin":"2009-11-25", "prioritaet": 7,"ort": u"Huecksenwagen","positionen": [{"menge": 12, "artnr": "14640/XL", "posnr": 1},],"versandeinweisungen": [{"guid": "2103839-XalE", "bezeichner": "avisierung48h","anweisung": "48h vor Anlieferung unter 0900-LOGISTIK avisieren"},]}}

    # __init__ gets called here. structure param is assigned example
    xml = dict2xml(example)
    xml.display()
