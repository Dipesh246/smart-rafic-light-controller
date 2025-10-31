DIRECTIONS = ["N", "E", "S", "W"]
LANES = ["straight", "left", "right"]

LANE_FLOW_RATES = {
    "Left": 2,  # Vehicles that can pass per second when green
    "Straight": 3,
    "Right": 1,
}

LANE_WEIGHTS = {
    "Left": 0.2,  # Less traffic typically
    "Straight": 0.6,  # Majority flow
    "Right": 0.2,
}