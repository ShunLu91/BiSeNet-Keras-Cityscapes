import numpy as np



# Cityscapes Colour map.
Cityscapes_Colors = [(128, 64, 128), (244, 35, 231), (69, 69, 69)
                # 0 = road, 1 = sidewalk, 2 = building
                ,(102, 102, 156), (190, 153, 153), (153, 153, 153)
                # 3 = wall, 4 = fence, 5 = pole
                ,(250, 170, 29), (219, 219, 0), (106, 142, 35)
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,(152, 250, 152), (69, 129, 180), (219, 19, 60)
                # 9 = terrain, 10 = sky, 11 = person
                ,(255, 0, 0), (0, 0, 142), (0, 0, 69)
                # 12 = rider, 13 = car, 14 = truck
                ,(0, 60, 100), (0, 79, 100), (0, 0, 230)
                # 15 = bus, 16 = train, 17 = motocycle
                ,(119, 10, 32)]
                # 18 = bicycle
