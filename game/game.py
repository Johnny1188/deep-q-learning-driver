import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (1200,150)
import math
import numpy as np
import sys
import pygame
from game.car import Car

np.random.seed(11)
pygame.init()

# Default environment parameters
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 640
NUM_OF_LANES = 6
NUM_OF_SEGMENTS_IN_LANE = 30
DIFFICULTY_OF_RIDE = 3 # Recommended
MAX_TIME = 40
LOOKAHEAD_DISTANCE = 5 # also change the input shape in the q_learner.py
LOOKBEHIND_DISTANCE = 0
LOOKTOSIDES_DISTANCE = 1
STATE_INPUT_DIM = ((2*LOOKTOSIDES_DISTANCE)+2)*(LOOKAHEAD_DISTANCE+3+LOOKBEHIND_DISTANCE)


class Road():
    def __init__(self,screen_w=SCREEN_WIDTH,screen_h=SCREEN_HEIGHT,segments_per_lane=NUM_OF_SEGMENTS_IN_LANE,num_of_lanes=NUM_OF_LANES,difficulty=DIFFICULTY_OF_RIDE,max_time=MAX_TIME):
        """
        Parameters
        ----------
        screen_w: integer
        Width of the screen (road) in pixels

        screen_h: integer
        Height of the screen (road) in pixels

        segment_per_lane: integer
        Number of segments in each lane = granularity of agent's moves forward and backward

        num_of_lanes: integer
        Road's number of lanes

        difficulty: integer
        Level of road difficulty - number of cars on the road. Must be in the range 0-10, inclusive.
        E.g. easiest = 0, hardest = 10

        max_time: integer
        Time after which to terminate the game automatically. 10 == 5 seconds (approximation)
        """
        # Initialize the blank screen
        self.w = screen_w
        self.h = screen_h
        self.screen = pygame.display.set_mode((self.w,self.h))

        # Initialize properties of the road
        self.num_of_lanes = num_of_lanes
        self.num_of_segments_per_lane = segments_per_lane
        self.one_lane_w = self.w // self.num_of_lanes
        self.height_of_segment = self.h // self.num_of_segments_per_lane
        self.road_shift = 0 # for road animation
        self.timer = .0
        self.step = 0
        self.difficulty = difficulty
        if self.difficulty not in range(0,11):
            raise ValueError(f"Level of difficulty must be an integer between 0 and 10 inclusive, your input is {difficulty}")
        
        self._setup_road_segments()
        self.car_size = [self.road_segments[0,0,0].width,(self.road_segments[0,0,0].height*3)+8]

        # Set the agent's car and surrounding cars
        agent_car_left_top_coordinates = (((self.num_of_lanes//2)*self.one_lane_w)+4,self.road_segments[0,int(math.floor(segments_per_lane*0.7)),0].y)
        self.agent = Car("game/imgs/green_car.png",lane_id=self.num_of_lanes//2,segment_id=int(math.floor(segments_per_lane*0.7)),speed=30,wanted_dims=self.car_size,left_top_coordinates=agent_car_left_top_coordinates)
        self.road_segments[self.agent.lane_id,self.agent.segment_id:self.agent.segment_id+3,1] = self.agent
        self.occupied_segments = [set(()) for _ in range(self.num_of_lanes)]
        self.cars = self._initialize_surrounding_cars()
        self.collapse = False
        self.max_time = max_time

    def _setup_road_segments(self):
        road_segments_list = []
        for lane_i in range(self.num_of_lanes):
            lane_segments = []
            for segment_in_lane in range(self.num_of_segments_per_lane):
                # Segment == (pygame.Rect(x,y,width,height),occupied-by-car[:boolean integer])
                lane_segments.append([pygame.Rect(
                    lane_i*self.one_lane_w+4,
                    segment_in_lane*self.height_of_segment+10,
                    self.one_lane_w-8,
                    self.height_of_segment-(self.height_of_segment//5)),None])
            road_segments_list.append(lane_segments)
        self.road_segments = np.array(road_segments_list,dtype=object)

    def _draw_dashed_lane(self, color, x_position, width=3, dash_length=10):
        for index in range(0, int(self.h//dash_length), 2):
            start = (x_position,(index * dash_length) + self.road_shift)
            end   = (x_position,((index + 1) * dash_length) + self.road_shift)
            pygame.draw.line(self.screen, color, start, end, width)
    
    def _draw_segments(self):
        for lane_segments in self.road_segments:
            for segment in lane_segments:
                color = (251, 221, 173) if segment[1] == None else (211, 84, 0)
                pygame.draw.rect(self.screen,color,segment[0])
    
    def _initialize_surrounding_cars(self):
        cars = []
        max_num_of_cars_per_lane = min(1,int(math.floor(self.difficulty/2) - round(self.difficulty/4.2)))

        for lane_i in range(self.num_of_lanes):
            num_of_cars = 0
            possible_segment_ids = set(np.where(self.road_segments[lane_i,:,1] == None)[0])

            while num_of_cars < max_num_of_cars_per_lane and len(possible_segment_ids) > 0:
                segment_id_to_place_car = np.random.choice([i for i in possible_segment_ids if i+1 in possible_segment_ids and i+2 in possible_segment_ids])
                
                car_left_top_coordinates = (self.road_segments[lane_i,segment_id_to_place_car,0].x,self.road_segments[lane_i,segment_id_to_place_car,0].y)
                new_car_to_place = Car(img_src="game/imgs/red_car.png",lane_id=lane_i,segment_id=segment_id_to_place_car,speed=np.random.randint(10,50),wanted_dims=self.car_size,left_top_coordinates=car_left_top_coordinates)

                for segment_chosen in range(3): # One car = 3 segments occupied
                    self.road_segments[lane_i,segment_id_to_place_car+segment_chosen,1] = new_car_to_place
                    self.occupied_segments[lane_i].add(segment_id_to_place_car+segment_chosen)
                    # Remove from possible_segment_ids - both segments in front of and behind the middle segment
                    possible_segment_ids.difference_update(*[[segment_id_to_place_car+segment_chosen,segment_id_to_place_car-segment_chosen]])

                cars.append( new_car_to_place )
                num_of_cars += 1
        
        return(cars)

    def _add_new_cars(self):
        for lane_id in np.random.choice(range(self.num_of_lanes),3,replace=False):
            anti_proba = (self.difficulty*2) + 6 + (sum([6 for segment_id in (-3,-2,-1,0,1,2,3,4) if segment_id in self.occupied_segments[max(0,lane_id-1)]])) + (sum([6 for segment_id in (-3,-2,-1,0,1,2,3,4) if segment_id in self.occupied_segments[min(self.num_of_lanes-1,lane_id+1)]]))
            add_new = True if np.random.randint(0,anti_proba) < self.difficulty else False
            if add_new and -1 not in self.occupied_segments[lane_id] and -2 not in self.occupied_segments[lane_id] and -3 not in self.occupied_segments[lane_id]:
                for segment_id in range(-3,0):
                    car_left_top_coordinates = (lane_id*self.one_lane_w+4,10 - (self.height_of_segment*3))
                    self.occupied_segments[lane_id].add(segment_id)
                self.cars.append( Car(img_src="game/imgs/red_car.png",lane_id=lane_id,segment_id=-3,speed=np.random.randint(10,60),wanted_dims=self.car_size,left_top_coordinates=car_left_top_coordinates) )

    def _wipe_out_car_from_road(self,car):
        for segment_id in range(car.segment_id,car.segment_id+3):
            if segment_id < self.num_of_segments_per_lane and segment_id >= 0:
                self.road_segments[car.lane_id,segment_id,1] = None
            if segment_id in self.occupied_segments[car.lane_id]: self.occupied_segments[car.lane_id].remove(segment_id)

    def _place_car_on_road(self,car,segment_id): # segment_id == the front of the car (=> segment_id+2 == the trunk of the car)
        for segment_id_occupied_by_car in range(segment_id,segment_id+3):
            if segment_id_occupied_by_car < self.num_of_segments_per_lane and segment_id_occupied_by_car >= 0:
                self.road_segments[car.lane_id,segment_id_occupied_by_car,1] = car
            self.occupied_segments[car.lane_id].add(segment_id_occupied_by_car)

    def _update_surrounding_cars_positions(self):
        for car in self.cars:
            # First, wipe out its last position
            self._wipe_out_car_from_road(car)

            new_car_y_position = car.car_rect.y + ((self.agent.speed-car.speed)*.025)

            if car.segment_id >= 0 and car.segment_id < self.num_of_segments_per_lane: # Still on the road with car's front segment
                if new_car_y_position > car.car_rect.y: # Moved down
                    new_car_segment_id = round(car.segment_id + abs((new_car_y_position-self.road_segments[car.lane_id,car.segment_id,0].y+10) / (self.height_of_segment)))
                    for new_segment_i in range(max(0,new_car_segment_id),min(self.num_of_segments_per_lane-1,new_car_segment_id+3)):
                        if self.road_segments[car.lane_id,new_segment_i,1] == self.agent: # Agent is too fast and he collapsed into the car in the front
                            self.collapse = True
                            return()
                else: # Moved up
                    new_car_segment_id = round(car.segment_id - abs(new_car_y_position-self.road_segments[car.lane_id,car.segment_id,0].y) / (self.height_of_segment+(self.height_of_segment//5)))
                    # Check whether the car needs to slow down in order to not cause collapse
                    for new_segment_id_to_be_occupied in range(max(0,new_car_segment_id),new_car_segment_id+3):
                        if new_segment_id_to_be_occupied < self.num_of_segments_per_lane and new_segment_id_to_be_occupied >= 0:
                            if self.road_segments[car.lane_id,new_segment_id_to_be_occupied,1].__class__ == Car:
                                car.speed = self.road_segments[car.lane_id,new_segment_id_to_be_occupied,1].speed
                                new_car_y_position = self.road_segments[car.lane_id,new_segment_id_to_be_occupied,1].car_rect.y + ((self.height_of_segment+(self.height_of_segment//5))*3)
                                new_car_segment_id = self.road_segments[car.lane_id,new_segment_id_to_be_occupied,1].segment_id + 3
                                break
                if np.random.randint(0,200) <= self.difficulty: # Randomly change car's lane
                    car = self._change_car_lane(car,new_car_segment_id)
            else: # Out of the road - still keep its position in memory
                if new_car_y_position > car.car_rect.y:
                    new_car_segment_id = [i for i in range(self.num_of_segments_per_lane-1) if self.road_segments[car.lane_id,i,0].y < new_car_y_position and self.road_segments[car.lane_id,i+1,0].y > new_car_y_position]
                    new_car_segment_id = new_car_segment_id[0] if len(new_car_segment_id) > 0 else round(car.segment_id + ((new_car_y_position-(car.segment_id*self.height_of_segment)) / self.height_of_segment))
                else:
                    new_car_segment_id = round(car.segment_id - (abs(new_car_y_position-car.car_rect.y) / self.height_of_segment))

            self._place_car_on_road(car,new_car_segment_id)

            car.car_rect.y = int(math.ceil(new_car_y_position))
            car.segment_id = new_car_segment_id
    
    def _change_car_lane(self,car,segment_id):
        possible_turns = [-1,1]
        np.random.shuffle(possible_turns)
        for possible_turn in possible_turns: # Left of right
            if (car.lane_id + possible_turn) >= self.num_of_lanes or car.lane_id + possible_turn < 0: # Out of road
                continue
            if len([vacant_place for vacant_place in self.road_segments[car.lane_id,segment_id:segment_id+3,1] if vacant_place.__class__ != Car]) == 3:
                # All three segments are vacant for the car to move there
                car.lane_id = car.lane_id + possible_turn
                car.car_rect.x = int(math.ceil(car.car_rect.x + (possible_turn*(self.car_size[0]+8))))
                break
        return(car)

    def _perform_car_action(self,car,action):
        new_lane_id = car.lane_id

        if action[0] == 1: # move left
            if car.lane_id-1 < 0:
                self.collapse = True
                return() # Out of the road
            if np.any(self.road_segments[car.lane_id-1,car.segment_id:car.segment_id+3,1]!=None):
                self.collapse = True
            else:
                self.road_segments[car.lane_id-1,car.segment_id:car.segment_id+3,1] = car
                self.road_segments[car.lane_id,car.segment_id:car.segment_id+3,1] = None
                new_lane_id = car.lane_id-1
                car.car_rect = car.car_rect.move((-self.one_lane_w,0))
        elif action[1] == 1: # move right
            if car.lane_id+1 >= self.num_of_lanes: # Out of the road
                self.collapse = True
                return()
            if np.any(self.road_segments[car.lane_id+1,car.segment_id:car.segment_id+3,1]!=None):
                self.collapse = True
            else:
                self.road_segments[car.lane_id+1,car.segment_id:car.segment_id+3,1] = car
                self.road_segments[car.lane_id,car.segment_id:car.segment_id+3,1] = None
                new_lane_id = car.lane_id+1
                car.car_rect = car.car_rect.move((self.one_lane_w,0))
        elif action[2] == 1: # Speed up
            car.speed = car.speed + 2
        elif action[3] == 1: # Slow down
            car.speed = car.speed - 2

        car.lane_id = new_lane_id

    def get_state(self):
        padded_road_w_zeros = np.pad(self.road_segments,((LOOKTOSIDES_DISTANCE,LOOKTOSIDES_DISTANCE),(LOOKAHEAD_DISTANCE,LOOKBEHIND_DISTANCE),(0,0)),mode="constant",constant_values=-10.0)

        visible_area_to_agent = padded_road_w_zeros[
            self.agent.lane_id : self.agent.lane_id+(LOOKTOSIDES_DISTANCE*2)+1,
            self.agent.segment_id : self.agent.segment_id+LOOKAHEAD_DISTANCE+LOOKBEHIND_DISTANCE+3,1
        ]

        def _assign_nums(x):
            if x == None: return(.0) # The segment is vaccant
            elif x == self.agent: return(1) # Occupied by the agent
            elif x.__class__ == Car and x != self.agent: return(10.) # Occupied by some surrounding car
            else: return(x) # Out of the road
        
        for lane_i in range(len(visible_area_to_agent)):
            visible_area_to_agent[lane_i] = [seg for seg in map(_assign_nums,visible_area_to_agent[lane_i])]

        # Part of the state is also the agent's current speed divided by 20 for stability purpose
        visible_area_to_agent = np.insert(visible_area_to_agent,0,[self.agent.speed/20]*visible_area_to_agent.shape[1],axis=0)
        return(visible_area_to_agent.astype(float))

    def play_step(self,action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        self.step += 1

        self._perform_car_action(self.agent,action)
        self._update_ui(True,True)

        if self.agent.speed < 300:
            if self.timer % 0.375 < 0.045:
                self._add_new_cars()
        elif self.agent.speed < 700:
            if self.timer % 0.2 < 0.045:
                self._add_new_cars()
        else:
            if self.timer % 0.1 < 0.045:
                self._add_new_cars()

        self._update_surrounding_cars_positions()

        """
        Reward
        ------
        EACH STEP WITHOUT AN ACCIDENT: reward = agent_speed / (maximum_time_lenght_of_road_run/6)
        WHEN ACCIDENT HAPPENDS: reward = -50
        """
        reward = ((self.agent.speed) / (self.max_time/6)) if self.collapse == False else -50

        if self.timer >= self.max_time:
            done = True
        else:
            done = self.collapse
            if done: self.agent.speed = 0

        return(reward,done)

    def _update_ui(self,show_cars=True,show_segments=True):
        self.timer += .025
        self.screen.fill((255,255,255))

        for lane_i in range(self.num_of_lanes):
            if lane_i >= 1: self._draw_dashed_lane((46, 49, 49),lane_i*self.one_lane_w,3,7)
        self.road_shift = (self.timer * self.agent.speed) % 6 # to animate speed

        if show_segments: self._draw_segments()

        if show_cars:
            for car in self.cars:
                self.screen.blit(self.agent.car,self.agent.car_rect)
                self.screen.blit(car.car,car.car_rect)

        pygame.display.flip()
