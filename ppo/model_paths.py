
NOWIND_MODEL = 'BEST/noWind/projectBEST'
LEFT_MODELS = [
    'BEST/left/ep690_1to2',
    'BEST/left/ep580_2to3',
    'BEST/left/ep640_3to4',
    'BEST/left/ep560_4to5'
]
GUSTYLEFT_MODELS = [
    'BEST/gustyLeft/ep530_1to2',
    'BEST/gustyLeft/ep390_2to3',
    'BEST/gustyLeft/ep680_3to4',
    'BEST/gustyLeft/ep690_4to5'
]
RIGHT_MODELS = [
    'BEST/right/ep680_1to2',
    'BEST/right/ep680_2to3',
    'BEST/right/ep670_3to4',
    'BEST/right/ep680_4to5'
]
GUSTYRIGHT_MODELS = [
    'BEST/gustyRight/ep670_1to2',
    'BEST/gustyRight/ep650_2to3',
    'BEST/gustyRight/ep670_3to4',
    'BEST/gustyRight/ep670_4to5'
]
SIDES_MODELS = [
    'BEST/sides/ep400_1to2',
    'BEST/sides/ep420_2to3',
    'BEST/sides/ep820_3to4',
    'BEST/sides/ep400_4to5'
]
GUSTYSIDES_MODELS = [
    'BEST/gustySides/ep830_1to2',
    'BEST/gustySides/ep1240_2to3',
    'BEST/gustySides/ep760_3to4',
    'BEST/gustySides/ep780_4to5'
]
ALL_MODELS = [
    NOWIND_MODEL,
    *LEFT_MODELS,
    *GUSTYLEFT_MODELS,
    *RIGHT_MODELS,
    *GUSTYRIGHT_MODELS,
    *SIDES_MODELS,
    *GUSTYSIDES_MODELS,
]
MODELS_INDEX = 0

ALLENVS = [ # (wind_wrapper, strength, name_for_1st_column_of_csv)
    (None, None, "noWind"),
    ("left", [0.1, 0.2], "left1to2"),
    ("left", [0.2, 0.3], "left2to3"),
    ("left", [0.3, 0.4], "left3to4"),
    ("left", [0.4, 0.5], "left4to5"),
    ("gustyLeft", [0.1, 0.2], "gustyLeft1to2"),
    ("gustyLeft", [0.2, 0.3], "gustyLeft2to3"),
    ("gustyLeft", [0.3, 0.4], "gustyLeft3to4"),
    ("gustyLeft", [0.4, 0.5], "gustyLeft4to5"),
    ("right", [0.1, 0.2], "right1to2"),
    ("right", [0.2, 0.3], "right2to3"),
    ("right", [0.3, 0.4], "right3to4"),
    ("right", [0.4, 0.5], "right4to5"),
    ("gustyRight", [0.1, 0.2], "gustyRight1to2"),
    ("gustyRight", [0.2, 0.3], "gustyRight2to3"),
    ("gustyRight", [0.3, 0.4], "gustyRight3to4"),
    ("gustyRight", [0.4, 0.5], "gustyRight4to5"),
    ("sides", [0.1, 0.2], "sides1to2"),
    ("sides", [0.2, 0.3], "sides2to3"),
    ("sides", [0.3, 0.4], "sides3to4"),
    ("sides", [0.4, 0.5], "sides4to5"),
    ("gustySides", [0.1, 0.2], "gustySides1to2"),
    ("gustySides", [0.2, 0.3], "gustySides2to3"),
    ("gustySides", [0.3, 0.4], "gustySides3to4"),
    ("gustySides", [0.4, 0.5], "gustySides4to5"),
]
ALLENVS_NAMES = [env[2] for env in ALLENVS]