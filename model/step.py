class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion

    def step(self):
        self.model.train()
        

class Validator:
    def __init__(self, model, dataloader, criterion):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion

    def step(self):
        self.model.eval()


def wrap_frame(frame, vector):
    return None