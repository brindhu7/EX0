from pattern import Checker
obj1 = Checker(250,25)
obj1.draw()
obj1.show()

from pattern import Circle
obj2 = Circle(1024, 200, (512, 256))
obj2.draw()
obj2.show()

from generator import ImageGenerator
gen = ImageGenerator("exercise_data", "Labels.json", 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
gen.show()