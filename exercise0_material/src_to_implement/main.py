from pattern import Checker
obj1 = Checker(250,25)
obj1.draw()
obj1.show()

from pattern import Circle
obj2 = Circle(1024, 200, (512, 256))
obj2.draw()
obj2.show()

from pattern import Spectrum
obj3 = Spectrum(255)
obj3.draw()
obj3.show()

from generator import ImageGenerator
obj4 = ImageGenerator('C:/Users/brind/PycharmProjects/EX0/exercise0_material/exercise_data/',
                      'C:/Users/brind/PycharmProjects/EX0/exercise0_material/Labels.json', 12, [32, 32, 3],
                      rotation=False, mirroring=False,
                             shuffle=False)

obj4.show()