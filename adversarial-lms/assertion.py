import hashlib
import json
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Assertion:
    story: str = ""
    sentence: str = ""
    subject: str = ""
    object: str = ""
    relation: str = ""
    general: bool = False
    additional: str = ""

    def __str__(self):
        if self.general:
            preface="Generally,"
        else:
            preface="Specifically,"

        str_rep = " ".join([preface,self.subject,self.relation,self.object+"."])
        return str_rep

    def __hash__(self):
        a = str(self)
        b = hashlib.md5(a.encode('utf-8')).hexdigest()
        return int(b, 16)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    @staticmethod
    def parse_from_string(assertion_string):
        a = Assertion()
        split_string = assertion_string.split("<")
        for split in split_string:
            if split == "":
                continue
            if "specific>" in split:
                a.general = False
            elif "general>" in split:
                a.general = True
            elif "subj>" in split:
                a.subject = split.replace("subj>","").strip()
            elif "/relation>" in split:
                continue
            elif "relation>" in split:
                a.relation = split.replace("relation>","").strip()
            elif "obj>" in split:
                a.object = split.replace("obj>","").strip()

        return a

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

