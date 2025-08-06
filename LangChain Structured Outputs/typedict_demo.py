from typing import TypedDict

class Person(TypedDict):
    name: str

person : Person = {"name": "Arham"}

print(person)