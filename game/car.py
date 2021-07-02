import pygame
import math

class Car():
    def __init__(self,img_src="game/imgs/red_car.png",lane_id=0,segment_id=0,speed=30,wanted_dims=[50,80],left_top_coordinates=(0,0)):
        self.lane_id = lane_id
        self.segment_id = segment_id
        self.speed = speed
        car_img_loaded_unresized = pygame.image.load(img_src)
        self.car_dims = self._find_out_target_car_dims(car_img_loaded_unresized,wanted_dims)
        self.car = pygame.transform.scale(car_img_loaded_unresized, (self.car_dims[0], self.car_dims[1]))
        self.car_rect = self.car.get_rect().move(left_top_coordinates)

    @staticmethod
    def _find_out_target_car_dims(img_loaded,wanted_dims):
        if wanted_dims[0] == None and wanted_dims[1] != None: # width missing, interpolate from the height and source img size
            wanted_dims[0] = int(math.floor(
                img_loaded.get_width()*(wanted_dims[1]/img_loaded.get_height())
                ))
        elif wanted_dims[1] == None and wanted_dims[0] != None: # height missing, interpolate from the width and source img size
            wanted_dims[1] = int(math.floor(
                img_loaded.get_height()*(wanted_dims[0]/img_loaded.get_width())
                ))
        elif (None,None) == wanted_dims:
            wanted_dims = [img_loaded.get_width(),img_loaded.get_height()]
        else:
            pass
        return(wanted_dims)