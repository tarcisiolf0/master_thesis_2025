class Nodule():
    def __init__(self, attenuation: str, edges : str, calcification: bool, localization: str, size: float, solid_component_size = float):
        self.attenuation = attenuation
        self.calcification = calcification
        self.edges = edges
        self.localization = localization
        self.size = float(size)
        self.solid_componet_size = float(solid_component_size)
    
    def show_info(self):
        print(f"Attenuation: {self.attenuation}")
        print(f"Calcification: {self.calcification}")
        print(f"Edges: {self.edges}")
        print(f"Localization: {self.localization}")
        print(f"Size: {self.size} mm")