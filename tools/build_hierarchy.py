import os, json, collections, timm
from nltk.corpus import wordnet as wn
d = os.path.join(os.path.dirname(timm.__file__), 'data', '_info')
wnids = open(f'{d}/imagenet_synsets.txt').read().split()
lemma = dict(l.rstrip('\n').split('\t', 1) for l in open(f'{d}/imagenet_synset_to_lemma.txt'))
assert len(wnids) == 1000

# (Level 1, Level 2, [WordNet synsets]) - first match wins, so specific groups come first
RULES = [
 ("Animal", "Dog",               ["dog.n.01", "wolf.n.01", "fox.n.01", "wild_dog.n.01", "hyena.n.01", "canine.n.02"]),
 ("Animal", "Cat",               ["feline.n.01"]),
 ("Animal", "Bear",              ["bear.n.01"]),
 ("Animal", "Primate",           ["primate.n.02"]),
 ("Animal", "Hoofed mammal",     ["ungulate.n.01"]),
 ("Animal", "Rodent & rabbit",   ["rodent.n.01", "lagomorph.n.01"]),
 ("Animal", "Sea mammal",        ["aquatic_mammal.n.01"]),
 ("Animal", "Other mammal",      ["mammal.n.01"]),
 ("Animal", "Bird",              ["bird.n.01"]),
 ("Animal", "Reptile",           ["reptile.n.01"]),
 ("Animal", "Amphibian",         ["amphibian.n.03"]),
 ("Animal", "Fish",              ["fish.n.01", "aquatic_vertebrate.n.01"]),
 ("Animal", "Insect",            ["insect.n.01"]),
 ("Animal", "Spider & arachnid", ["arachnid.n.01"]),
 ("Animal", "Crustacean",        ["crustacean.n.01"]),
 ("Animal", "Other invertebrate",["invertebrate.n.01"]),
 ("Animal", "Other animal",      ["animal.n.01"]),
 ("Food", "Fruit",               ["edible_fruit.n.01", "fruit.n.01"]),
 ("Food", "Vegetable",           ["vegetable.n.01"]),
 ("Food", "Drink",               ["beverage.n.01"]),
 ("Food", "Bread & dessert",     ["baked_goods.n.01", "dessert.n.01", "frozen_dessert.n.01", "sweet.n.03"]),
 ("Food", "Dish & meal",         ["dish.n.02", "food.n.02", "food.n.01", "foodstuff.n.02"]),
 ("Plant & fungus", "Fungus",    ["fungus.n.01"]),
 ("Plant & fungus", "Plant & flower", ["plant.n.02", "plant_part.n.01"]),
 ("Person", "Person",            ["person.n.01"]),
 ("Nature & place", "Landscape", ["geological_formation.n.01"]),
 ("Vehicle", "Car & truck",      ["motor_vehicle.n.01", "car.n.01"]),
 ("Vehicle", "Boat & ship",      ["vessel.n.02", "boat.n.01"]),
 ("Vehicle", "Aircraft",         ["aircraft.n.01", "spacecraft.n.01"]),
 ("Vehicle", "Train",            ["train.n.01", "locomotive.n.01", "streetcar.n.01", "passenger_car.n.01"]),
 ("Vehicle", "Bicycle & cart",   ["wheeled_vehicle.n.01"]),
 ("Vehicle", "Other vehicle",    ["conveyance.n.03", "vehicle.n.01"]),
 ("Sports & leisure", "Sports equipment", ["sports_equipment.n.01", "ball.n.01", "game_equipment.n.01", "toy.n.01", "toy.n.03", "plaything.n.01"]),
 ("Device & tool", "Musical instrument", ["musical_instrument.n.01"]),
 ("Device & tool", "Computer & electronics", ["computer.n.01", "electronic_equipment.n.01", "telephone.n.01", "peripheral.n.01", "display.n.06", "screen.n.03", "communication_system.n.01"]),
 ("Device & tool", "Home appliance", ["home_appliance.n.01", "kitchen_appliance.n.01"]),
 ("Device & tool", "Weapon",     ["weapon.n.01"]),
 ("Device & tool", "Tool",       ["tool.n.01", "hand_tool.n.01", "implement.n.01"]),
 ("Device & tool", "Machine & device", ["machine.n.01", "device.n.01", "equipment.n.01", "instrument.n.01", "apparatus.n.01", "mechanism.n.05"]),
 ("Clothing & accessory", "Clothing", ["clothing.n.01", "garment.n.01", "footwear.n.02", "headdress.n.01"]),
 ("Clothing & accessory", "Accessory", ["accessory.n.01", "jewelry.n.01", "adornment.n.01", "covering.n.02", "protective_garment.n.01", "bag.n.04", "bag.n.01"]),
 ("Household", "Furniture", ["furniture.n.01", "furnishing.n.02"]),
 ("Household", "Container & tableware", ["container.n.01", "vessel.n.03", "tableware.n.01", "crockery.n.01", "cutlery.n.02"]),
 ("Building & street", "Building & structure", ["structure.n.01", "establishment.n.04", "building.n.01", "facility.n.01", "area.n.05"]),
 ("Household", "Toiletries & cosmetics", ["toiletry.n.01", "cosmetic.n.01", "drug.n.01"]),
 ("Household", "Fabric & linen", ["fabric.n.01", "linen.n.03", "piece_of_cloth.n.01", "cushion.n.03", "padding.n.01", "paper.n.01"]),
 ("Building & street", "Sign & signal", ["sign.n.02", "signal.n.01"]),
]
cache = {}
def closure(s):
    out = set()
    for path in s.hypernym_paths(): out.update(p.name() for p in path)
    return out
classes, unmapped = [], []
for i, w in enumerate(wnids):
    s = wn.synset_from_pos_and_offset('n', int(w[1:]))
    hyp = closure(s)
    name = lemma[w].split(',')[0].strip().replace('_', ' ')
    l1 = l2 = None
    for a, b, syns in RULES:
        if any(x in hyp for x in syns): l1, l2 = a, b; break
    if l1 is None:
        l1, l2 = ("Object", "Other object"); unmapped.append((i, name, [p.name() for p in s.hypernym_paths()[0]][-4:]))
    classes.append({"index": i, "name": name[:1].upper() + name[1:], "level1": l1, "level2": l2})
c1 = collections.Counter(c["level1"] for c in classes); c2 = collections.Counter((c["level1"], c["level2"]) for c in classes)
print(len(c1), "level-1 groups:", dict(c1)); print(len(c2), "level-2 groups")
for k, v in sorted(c2.items()): print(f"  {k[0]:22s} {k[1]:26s} {v}")
print("unmapped:", len(unmapped)); [print("  ", u) for u in unmapped[:60]]
json.dump({"levels": ["Group", "Subgroup", "Class"], "classes": classes}, open("RapidNet_ImageNet_Hierarchy.json", "w"), indent=0)
