import os
import sys

dataset = sys.argv[1]
people = set([])


with open(dataset+"/"+"mtest.txt", "w") as ptrain:
    with open(dataset+"/"+ "train.txt") as train:
        for line in train:
            e1, r, e2 = line.strip().split()
            if r == "/people/person/profession":
                people.add(e1)
                ptrain.write(line)


