class Dataset:
    def __init__(self):
        self.vehicle_jam = dict()
        self.stations = dict()
        self.used_vehicle = set()  # refresh before assigning
        # from json
        self.map_info = None
        self.map_info_unchanged = None
        self.original_map_info = None
        self.all_routes = None
        self.all_stations = None
        self.stations_name = None
        self.adjacent_matrix = None
        self.valued_adjacent_matrix = None
        self.bays_relation = None
        # from db
        self.orders = dict()
        self.vehicles = dict()
        self.tracks = dict()
        # database cursor
        self.db_connection = None
        self.db_cursor = None
        # not used
        self.task_data = None
        self.machine_location = None

        # control on-off
        self.pattern = 1  # 1:circulating; 0:just one time

    class Control:
        def __init__(self):
            self.num_vehicle = 0
            self.global_start_time = None
            self.task_num = 9999999
            self.build_map = False
            # self.build_map = True
            self.out_path = 1  # 1:to_excel; 0:to_db


    class Task:
        def __init__(self):
            self.id = None
            self.start_location = None
            self.end_location = None
            self.start_time = None
            self.end_time = None
            self.real_start_time = None
            self.pick_route = []
            self.delivery_route = None
            self.vehicle_assigned = None
            self.finished = 0
            self.handling_time = 10

    class Stations:
        def __init__(self):
            self.location = None
            self.position = None
            self.id = None
            self.name = None
            self.left_node = None
            self.right_node = None

    class Vehicle:
        def __init__(self):
            self.start_location = None
            self.id = None
            self.speed = 800
            self.status = 0
            self.available_time = 0
            self.job_load = []
            self.bays = None

    class Route:
        def __init__(self):
            self.id = None
            self.available_time = 0
            self.capacity = 5
            self.vehicle_count = 0
            self.record = []

    class TimeInterval:
        def __init__(self, start_time, end_time):
            self.start = start_time
            self.end = end_time
