from scared import Experiment
from config import config_ingredient
ex = Experiment('Paper Abstract manipulation with BERT', ingredients=[config_ingredient])

@ex.config
def set_seed():
    seed = 1234

@config_ingredient.capture
def 

@ex.automain
def run(cfg):
