# Copyright (c) Facebook, Inc. and its affiliates.
from minihack import MiniHackSkill
from minihack.envs import register
from nle import nethack


class MiniHackLCLevitatePotionPickup(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------------
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
-------------
ENDMAP
REGION:(0,0,12,6),lit,"ordinary"
$left_bank = selection:fillrect (1,1,5,5)
$right_bank = selection:fillrect (7,1,11,5)
OBJECT:('!',"levitation"),rndcoord($left_bank),blessed
BRANCH:(1,1,5,5),(0,0,0,0)
STAIR:rndcoord($right_bank),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)


class MiniHackLCLevitatePotionPickupRestrictedActions(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)

        ACTIONS = tuple(nethack.CompassDirection) + (nethack.Command.PICKUP, \
                                                     nethack.Command.QUAFF, \
                                                     nethack.Command.FIRE)
        kwargs["actions"] = ACTIONS

        
        des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------------
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
-------------
ENDMAP
REGION:(0,0,12,6),lit,"ordinary"
$left_bank = selection:fillrect (1,1,5,5)
$right_bank = selection:fillrect (7,1,11,5)
OBJECT:('!',"levitation"),rndcoord($left_bank),blessed
BRANCH:(1,1,5,5),(0,0,0,0)
STAIR:rndcoord($right_bank),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)
        


class MiniHackLCLevitatePotionInv(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        kwargs["autopickup"] = kwargs.pop("autopickup", True)
        des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------------
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
-------------
ENDMAP
REGION:(0,0,12,6),lit,"ordinary"
$right_bank = selection:fillrect (7,1,11,5)
OBJECT:('!',"levitation"),(2,2),blessed
BRANCH:(2,2,2,2),(0,0,0,0)
STAIR:rndcoord($right_bank),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)




class MiniHackLCLevitatePotionInvRestrictedActions(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        kwargs["autopickup"] = kwargs.pop("autopickup", True)

        ACTIONS = tuple(nethack.CompassDirection) + (nethack.Command.PICKUP, \
                                                     nethack.Command.QUAFF, \
                                                     nethack.Command.FIRE)
        kwargs["actions"] = ACTIONS
        
        des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------------
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
-------------
ENDMAP
REGION:(0,0,12,6),lit,"ordinary"
$right_bank = selection:fillrect (7,1,11,5)
OBJECT:('!',"levitation"),(2,2),blessed
BRANCH:(2,2,2,2),(0,0,0,0)
STAIR:rndcoord($right_bank),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)




        
class MiniHackLCLevitateRingPickup(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------------
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
-------------
ENDMAP
REGION:(0,0,12,6),lit,"ordinary"
$left_bank = selection:fillrect (1,1,5,5)
$right_bank = selection:fillrect (7,1,11,5)
OBJECT:('=',"levitation"),rndcoord($left_bank),blessed
BRANCH:(1,1,5,5),(0,0,0,0)
STAIR:rndcoord($right_bank),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)


class MiniHackLCLevitateRingInv(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        kwargs["autopickup"] = kwargs.pop("autopickup", True)
        des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------------
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
-------------
ENDMAP
REGION:(0,0,12,6),lit,"ordinary"
$right_bank = selection:fillrect (7,1,11,5)
OBJECT:('=',"levitation"),(2,2),blessed
BRANCH:(2,2,2,2),(0,0,0,0)
STAIR:rndcoord($right_bank),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)


class MiniHackLCLevitate(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------------
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
|.....L.....|
-------------
ENDMAP
REGION:(0,0,12,6),lit,"ordinary"
$left_bank = selection:fillrect (1,1,5,5)
$right_bank = selection:fillrect (7,1,11,5)
IF [33%] {
    OBJECT:('!',"levitation"),rndcoord($left_bank),blessed
} ELSE {
    IF [50%] {
        OBJECT:('=',"levitation"),rndcoord($left_bank),blessed
    } ELSE {
        OBJECT:('[',"levitation boots"),rndcoord($left_bank),blessed
    }
}
BRANCH:(1,1,5,5),(0,0,0,0)
STAIR:rndcoord($right_bank),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)


class MiniHackLC(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file="lava_crossing.des", **kwargs)


register(
    id="MiniHack-LavaCross-Levitate-Potion-Pickup-v0",
    entry_point="minihack.envs.skills_lava:MiniHackLCLevitatePotionPickup",
)
register(
    id="MiniHack-LavaCross-Levitate-Potion-Pickup-v1",
    entry_point="minihack.envs.skills_lava:MiniHackLCLevitatePotionPickupRestrictedActions",
)

register(
    id="MiniHack-LavaCross-Levitate-Potion-Inv-v0",
    entry_point="minihack.envs.skills_lava:MiniHackLCLevitatePotionInv",
)
register(
    id="MiniHack-LavaCross-Levitate-Potion-Inv-v1",
    entry_point="minihack.envs.skills_lava:MiniHackLCLevitatePotionInvRestrictedActions",
)
register(
    id="MiniHack-LavaCross-Levitate-Ring-Pickup-v0",
    entry_point="minihack.envs.skills_lava:MiniHackLCLevitateRingPickup",
)
register(
    id="MiniHack-LavaCross-Levitate-Ring-Inv-v0",
    entry_point="minihack.envs.skills_lava:MiniHackLCLevitateRingInv",
)
register(
    id="MiniHack-LavaCross-Levitate-v0",
    entry_point="minihack.envs.skills_lava:MiniHackLCLevitate",
)
register(
    id="MiniHack-LavaCross-v0",
    entry_point="minihack.envs.skills_lava:MiniHackLC",
)
