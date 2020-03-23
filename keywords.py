from typing import List, Optional
import psytrance

category: "social"

shift_topic_corona_de = ["corona", "virus","pandemie","rki",]

n = 90 # we probably want like 900 + 124 control in the end
Nctrl = 10 # probably like ~100

# control keyword google trend data curve smoothness and slope params:
gold_slope = 0.01
gold_smooth = 0.8
#^not sure what math to use, dont know if higher or lower is better for smooth param
desparation_rate = 1.2 #loosen factor for above criteria


def generate_bow(topic:str=category,n:int=N) -> List[str]:
    """
    probably use pretrained glove by stanford to find topk similar and check POS
    """
    raise NotImplementedError

def top_searches_but_avoid(topic:str=category,n:int=N) -> List[str]:
    """
    same interface as generate_bow, opposite intent:
    start out from all time frequent searches
    with return of psytrance.top_searches_no_outlier()
    then use comparison metric used in generate_bow to filter
    out words similar to param topic

    """

    raise NotImplementedError



def check_trend_unchanging(kw:str, time_data:np.ndarray, max_abs_slope=gold_slope, min_smoothness=gold_smooth) -> bool:

    """
    Uses math
    to check wether the time_data curve is

    1. horizontal enough
    2. smooth enough

    returns True if both are satisfied, False otherwise

    """
    raise NotImplementedError


def generate_control(topic:str=category,n:int=Nctrl,
                     init_abs_slope=gold_slope,
                     init_smoothness=gold_smooth,
                     desparation_rate=desparation_rate) -> List[str]:
    """
    Generate a diverse group of control keywords unrelated to the topic
    whose search volume changes minimally over time

    start with desired curve properties and despair over time to give in to more lax values if no
    keywords with these properties can be found

    this is needed for the network to have an absolute reference

    e.g. 'fb login', 'text', 'tankstelle','kaffee', 'bruh'

    """
    approved_control_group = []

    funding_factor = 5

    curr_bound = init_abs_slope
    curr_smooth = init_smoothness
    desparation_inverse = 1./desparation #depends on math used for smoothness checking

    while len(approved) < n:

        candidate_controls = top_searches_but_avoid(kw,funding_factor*n) #TODO top_searches should use
        #psytrance.top_searches_no_outlier and filter out similar ones to the keyword

        for candidate in candidate_controls:

            approval =\
            check_trend_unchanging(candidate,max_abs_slope=curr_bound,min_smoothness=curr_smooth)
            if approval:
                approved_control_group.append(candidate)

        #if we go to next while iter, we failed to find enough candidates
        #lets loosen our criteria a little:
        curr_bound *= desparation_rate
        curr_smooth *= desparation_inverse #depends on math

    approved_control_group = approved_control_group[:n-1] #should probably just avoid overgeneration
    # in last for while loop by breaking both loops at once (can this be done?)

    return approved_control_group

def generate_features(topic:str=category,n:int=N, nctrl:int=Nctrl) -> List[str]:
    """
    generates keyword features for our neural net
    this should probably be the only function in this module seen by the rest of the package
    TODO encapsulation stuff

    """

    return features 
