rels = ["oEffect", "oReact", "oWant", "xAttr", "xEffect",
        "xIntent","xNeed","xReact","xWant"]

text_rels = ["has the effect on others of [FILL]",
             "makes others react [FILL]",
             "makes others want to do [FILL]",
             "described as [FILL]",
             "has the effect [FILL]",
             "causes the event because [FILL]",
             "before needs to [FILL]",
             "after feels [FILL]",
             "after wants to [FILL]"]

rels = ["AtLocation",
 "CapableOf",
 "Causes",
 "CausesDesire",
 "CreatedBy",
 "Desires",
"HasA",
 "HasFirstSubevent",
 "HasLastSubevent",
 "HasPrerequisite",
 "HasProperty",
 "HasSubEvent",
 "HinderedBy",
 "InstanceOf",
 "isAfter",
"isBefore",
"isFilledBy",
"MadeOf",
 "MadeUpOf",
 "MotivatedByGoal",
 "NotDesires",
 "ObjectUse",
 "UsedFor",
 "oEffect",
"oReact",
"oWant",
"PartOf",
 "ReceivesAction",
 "xAttr",
"xEffect",
 "xIntent",
 "xNeed",
 "xReact",
 "xReason",
 "xWant"]

text_rels = ["located or found at/in/on",
             "is/are capable of",
"causes",
"makes someone want",
"is created by",
"desires",
"has, possesses or contains",
"begins with the event/action",
"ends with the event/action",
"to do this, one requires",
"can be characterized by being/having","includes the event/action",
"can be hindered by",
"is an example/instance of",
"happens after",
"happens before",
"blank can be filled by",
"is made of",
"made (up) of",
"is a step towards accomplishing the goal","do(es) not desire",
"used for","used for",
"has the effect on others of",
"makes others react",
"makes others want to do",
"is a part of",
"can receive or be affected by the action",
 "described as",
"has the effect",
"causes the event because",
"before needs to",
"after feels",
"because",
"after wants to"]

glucose_rel_map = {
    "Causes/Enables":"causes or enables",
    "Enables":"enables",
    "Results in":"results in",
    "Motivates":"motivates",
}

atomic_rel_map = dict(zip(rels,text_rels))

relation_map = {
        "HasFirstSubevent":"has the first sub event",
        "HasLastSubevent":"has the last sub event",
        "FormOf": "is a form of",
        "IsA": "is a",
        "NotDesires":"does not desire",
        "RelatedTo": "is related to",
        "HasProperty": "has the property",
        "HasContext": "has the context",
        "DerivedFrom": "is derived from",
        "DefinedAs":"is defined as",
        "UsedFor": "is used for",
        "Causes": "causes",
        "Synonym": "is a synonym of",
        "Antonym": "is a antonym of",
        "CapableOf": "is capable of",
        "HasA": "has a",
        "Desires": "desires",
        "AtLocation": "is located at",
        "ReceivesAction": "receives the action",
        "SimilarTo": "is similar to",
        "CausesDesire": "causes a desire for",
        "DistinctFrom": "is distinct from",
        "PartOf": "is a part of",
        "HasSubevent": "has the subevent",
        "HasPrerequisite": "has the prerequisite",
        "MannerOf": "is a manner of",
        "MotivatedByGoal": "is motivated by the goal",
        "MadeOf":"is made of",
    "NotIsA": "is not a",
    "InstanceOf":"is an instance of",
    "SymbolOf":"is a symbol of",
    "CreatedBy":"is created by",
    "NotCapableOf":"is not capable of",
    "NotHasProperty":"does not have the property"
}
relation_map.update(atomic_rel_map)
relation_map.update(glucose_rel_map)
tokens = {
    "story_start": "<story>",
    "story_end": "</story>",
    "relation_start": "<relation>",
    "relation_end": "</relation>",
    "sentence_start": "<sentence>",
    "sentence_end": "</sentence>",
    "general": "<general>",
    "specific": "<specific>"
}
additional_tokens = ["PersonX", "PersonY",
                     "Something_A", "Something_B", "Something_C", "Something_D", "Something_E",
                     "People_A", "People_B", "People_C", "People_D", "People_E",
                                                                     "Someone_A", "Someone_B", "Someone_C", "Someone_D",
                     "Someone_E",
                     "Some_Event_A", "Some_Event_B", "Some_Event_C", "<end>", "<subj>", "<obj>"]
